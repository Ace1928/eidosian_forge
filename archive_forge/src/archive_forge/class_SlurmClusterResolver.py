import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.cluster_resolver.SlurmClusterResolver')
class SlurmClusterResolver(ClusterResolver):
    """ClusterResolver for system with Slurm workload manager.

  This is an implementation of ClusterResolver for Slurm clusters. This allows
  the specification of jobs and task counts, number of tasks per node, number
  of GPUs on each node and number of GPUs for each task. It retrieves system
  attributes by Slurm environment variables, resolves allocated computing node
  names, constructs a cluster and returns a ClusterResolver object which can be
  used for distributed TensorFlow.
  """

    def __init__(self, jobs=None, port_base=8888, gpus_per_node=None, gpus_per_task=None, tasks_per_node=None, auto_set_gpu=True, rpc_layer='grpc'):
        """Creates a new SlurmClusterResolver object.

    For any parameter not set it will query the environment for the value.
    It uses those parameters to check which nodes have processes reside on and
    resolves their hostnames.
    With the number tasks per node it offsets the port number for each process.
    With the number of GPUs per node and per task it allocates GPUs to tasks by
    setting environment variables.
    Using the resolver works best (and is easier) with homogeneous tasks but
    heterogeneous tasks (number of tasks varying per node) are also possible as
    long as the number of GPUs per task stays constant.

    Used environment variables:
      - SLURM_PROCID
      - (opt) SLURM_STEP_NUM_TASKS
      - (opt) SLURM_STEP_NODELIST
      - (opt) SLURM_STEP_TASKS_PER_NODE

    Args:
      jobs: Dictionary with job names as key and number of tasks in the job as
        value. Defaults to as many 'worker's as there are (Slurm) tasks.
      port_base: The first port number to start with for processes on a node.
      gpus_per_node: Number of GPUs available on each node. Defaults to the
        number of GPUs reported by nvidia-smi
      gpus_per_task: Number of GPUs to be used for each task. Default is to
        evenly distribute the gpus_per_node to tasks_per_node.
      tasks_per_node: Number of tasks running on each node. Can be an integer if
        the number of tasks per node is constant or a dictionary mapping
        hostnames to number of tasks on that node. If not set the Slurm
        environment is queried for the correct mapping.
      auto_set_gpu: Set the visible CUDA devices automatically while resolving
        the cluster by setting CUDA_VISIBLE_DEVICES environment variable.
        Defaults to True.
      rpc_layer: The protocol TensorFlow used to communicate between nodes.
        Defaults to 'grpc'.

    Returns:
      A ClusterResolver object which can be used with distributed TensorFlow.

    Raises:
      RuntimeError: If requested more GPUs per node than available or
        requested more tasks than assigned tasks or
        resolving missing values from the environment failed.
    """
        self._rank = self._resolve_own_rank()
        if jobs is None:
            jobs = {'worker': self._resolve_num_tasks()}
        self._jobs = jobs
        self._port_base = port_base
        if tasks_per_node is None:
            self._task_configuration = self._resolve_task_configuration()
        elif isinstance(tasks_per_node, dict):
            self._task_configuration = tasks_per_node
        else:
            hostlist = self._resolve_hostlist()
            self._task_configuration = {host: int(tasks_per_node) for host in hostlist}
        max_tasks_per_node = max(self._task_configuration.values())
        num_tasks = sum(self._task_configuration.values())
        if gpus_per_node is None:
            gpus_per_node = get_num_gpus()
        if gpus_per_task is None:
            gpus_per_task = gpus_per_node // max_tasks_per_node
        self._gpus_per_node = gpus_per_node
        self._gpus_per_task = gpus_per_task
        self._auto_set_gpu = auto_set_gpu
        self.task_type = None
        self.task_id = None
        self.rpc_layer = rpc_layer
        self._gpu_allocation = []
        self._cluster_allocation = {}
        if max_tasks_per_node * self._gpus_per_task > self._gpus_per_node:
            raise RuntimeError('Requested more GPUs per node than available.')
        if sum(self._jobs.values()) != num_tasks:
            raise RuntimeError('Requested {} tasks but only {} were assigned.'.format(sum(self._jobs.values()), num_tasks))

    def _resolve_own_rank(self):
        """Returns the rank of the current task in range [0, num_tasks)."""
        return int(_get_slurm_var('PROCID'))

    def _resolve_num_tasks(self):
        """Returns the number of tasks for the current job step."""
        return _get_num_slurm_tasks()

    def _resolve_hostlist(self):
        """Returns a list of hostnames for nodes running the current job step."""
        return expand_hostlist(_get_slurm_var('STEP_NODELIST'))

    def _resolve_task_configuration(self):
        """Creates a mapping of hostnames to the number of tasks allocated on it.

    Reads the SLURM environment to determine the nodes involved in the current
    job step and number of tasks running on each node.

    Returns a dictionary mapping each hostname to the number of tasks.
    """
        hostlist = self._resolve_hostlist()
        tasks_per_node = expand_tasks_per_node(_get_slurm_var('STEP_TASKS_PER_NODE'))
        return {host: num_tasks for host, num_tasks in zip(hostlist, tasks_per_node)}

    def cluster_spec(self):
        """Returns a ClusterSpec object based on the latest instance group info.

    This returns a ClusterSpec object for use based on information from the
    specified initialization parameters and Slurm environment variables. The
    cluster specification is resolved each time this function is called. The
    resolver extract hostnames of nodes by scontrol and pack tasks in that
    order until a node a has number of tasks that is equal to specification.
    GPUs on nodes are allocated to tasks by specification through setting
    CUDA_VISIBLE_DEVICES environment variable.

    Returns:
      A ClusterSpec containing host information retrieved from Slurm's
        environment variables.
    """
        task_list = []
        self._gpu_allocation = []
        self._cluster_allocation = {}
        for host, num_tasks in sorted(self._task_configuration.items()):
            for port_offset, gpu_offset in zip(range(num_tasks), range(0, self._gpus_per_node, self._gpus_per_task)):
                host_addr = '%s:%d' % (host, self._port_base + port_offset)
                task_list.append(host_addr)
                gpu_id_list = []
                for gpu_id in range(gpu_offset, gpu_offset + self._gpus_per_task):
                    gpu_id_list.append(str(gpu_id))
                self._gpu_allocation.append(','.join(gpu_id_list))
        cluster_rank_offset_start = 0
        cluster_rank_offset_end = 0
        for task_type, num_tasks in sorted(self._jobs.items()):
            cluster_rank_offset_end = cluster_rank_offset_start + num_tasks
            self._cluster_allocation[task_type] = task_list[cluster_rank_offset_start:cluster_rank_offset_end]
            if cluster_rank_offset_start <= self._rank < cluster_rank_offset_end:
                self.task_type = task_type
                self.task_id = self._rank - cluster_rank_offset_start
            cluster_rank_offset_start = cluster_rank_offset_end
        if self._auto_set_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_allocation[self._rank]
        return ClusterSpec(self._cluster_allocation)

    def get_task_info(self):
        """Returns job name and task_id for the process which calls this.

    This returns the job name and task index for the process which calls this
    function according to its rank and cluster specification. The job name and
    task index are set after a cluster is constructed by cluster_spec otherwise
    defaults to None.

    Returns:
      A string specifying job name the process belongs to and an integer
        specifying the task index the process belongs to in that job.
    """
        return (self.task_type, self.task_id)

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        """Returns the master string for connecting to a TensorFlow master.

    Args:
      task_type: (Optional) Overrides the default auto-selected task type.
      task_id: (Optional) Overrides the default auto-selected task index.
      rpc_layer: (Optional) Overrides the default RPC protocol TensorFlow uses
        to communicate across nodes.

    Returns:
      A connection string for connecting to a TensorFlow master.
    """
        task_type = task_type if task_type is not None else self.task_type
        task_id = task_id if task_id is not None else self.task_id
        if task_type is not None and task_id is not None:
            return format_master_url(self.cluster_spec().task_address(task_type, task_id), rpc_layer or self.rpc_layer)
        return ''

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        del task_type, task_id, config_proto
        return {'GPU': self._gpus_per_task}