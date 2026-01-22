import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
class _WorkerContext(object):
    """The worker context class.

  This context object provides configuration information for each task. One
  context manager with a worker context object will be created per
  invocation to the `worker_fn` where `get_current_worker_context` can be called
  to access the worker context object.
  """

    def __init__(self, strategy, cluster_spec, task_type, task_id, session_config=None, rpc_layer='grpc', worker_barrier=None):
        """Initialize the worker context object.

    Args:
      strategy: a `DistributionStrategy` object.
      cluster_spec: a ClusterSpec object. It can be empty or None in the local
        training case.
      task_type: a string indicating the role of the corresponding task, such as
        "worker" or "ps". It can be None if it is local training or in-graph
        replicated training.
      task_id: an integer indicating id of the corresponding task. It can be
        None if it is local training or in-graph replicated training.
      session_config: an optional `tf.compat.v1.ConfigProto` object.
      rpc_layer: optional string specifying the RPC protocol for communication
        with worker masters. If None or empty, hosts in the `cluster_spec` will
        be used directly.
      worker_barrier: optional, the barrier object for worker synchronization.
    """
        self._strategy = strategy
        self._cluster_spec = cluster_spec
        self._task_type = task_type
        self._task_id = task_id
        self._session_config = session_config
        self._worker_barrier = worker_barrier
        self._rpc_layer = rpc_layer
        self._master_target = self._get_master_target()
        self._num_workers = _get_num_workers(cluster_spec)
        self._is_chief_node = self._is_chief()

    def _debug_message(self):
        if self._cluster_spec:
            return '[cluster_spec: %r, task_type: %r, task_id: %r]' % (self._cluster_spec, self.task_type, self.task_id)
        else:
            return '[local]'

    def __enter__(self):
        old_context = get_current_worker_context()
        if old_context:
            raise ValueError('You cannot run distribute coordinator in a `worker_fn`.\t' + self._debug_message())
        _worker_context.current = self

    def __exit__(self, unused_exception_type, unused_exception_value, unused_traceback):
        _worker_context.current = None

    def _get_master_target(self):
        """Return the master target for a task."""
        if not self._cluster_spec or self._task_type == _TaskType.EVALUATOR:
            return ''
        if not self._task_type:
            if _TaskType.CHIEF in self._cluster_spec.jobs:
                task_type = _TaskType.CHIEF
                task_id = 0
            else:
                assert _TaskType.WORKER in self._cluster_spec.jobs
                task_type = _TaskType.WORKER
                task_id = 0
        else:
            task_type = self._task_type
            task_id = self._task_id
        prefix = ''
        if self._rpc_layer:
            prefix = self._rpc_layer + '://'
        return prefix + self._cluster_spec.job_tasks(task_type)[task_id or 0]

    def _is_chief(self):
        """Return whether the task is the chief worker."""
        if not self._cluster_spec or self._task_type in [_TaskType.CHIEF, _TaskType.EVALUATOR, None]:
            return True
        if _TaskType.CHIEF not in self._cluster_spec.jobs and self._task_type == _TaskType.WORKER and (self._task_id == 0):
            return True
        return False

    def wait_for_other_workers(self):
        """Waits for other workers to reach the same call to this method.

    Raises:
      ValueError: if `worker_barrier` is not passed to the __init__ method.
    """
        if not self._worker_barrier:
            return
        self._worker_barrier.wait()

    def session_creator(self, scaffold=None, config=None, checkpoint_dir=None, checkpoint_filename_with_path=None, max_wait_secs=7200):
        """Returns a session creator.

    The returned session creator will be configured with the correct master
    target and session configs. It will also run either init ops or ready ops
    by querying the `strategy` object when `create_session` is called on it.

    Args:
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      config: `ConfigProto` proto used to configure the session.
      checkpoint_dir: A string. Optional path to a directory where to restore
        variables.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
        Only one of `checkpoint_dir` or `checkpoint_filename_with_path` can be
        specified.
      max_wait_secs: Maximum time to wait for the session to become available.

    Returns:
      a descendant of SessionCreator.
    """
        if config:
            session_config = copy.deepcopy(config)
            session_config.MergeFrom(self._session_config)
        else:
            session_config = self._session_config
        if not self._strategy or self._strategy.extended.experimental_should_init:
            logging.info('Creating chief session creator with config: %r', config)
            return monitored_session.ChiefSessionCreator(scaffold, master=self.master_target, config=session_config, checkpoint_dir=checkpoint_dir, checkpoint_filename_with_path=checkpoint_filename_with_path)
        else:
            logging.info('Creating worker session creator with config: %r', config)
            return monitored_session.WorkerSessionCreator(scaffold, master=self.master_target, config=session_config, max_wait_secs=max_wait_secs)

    @property
    def session_config(self):
        return copy.deepcopy(self._session_config)

    @property
    def has_barrier(self):
        """Whether the barrier is set or not."""
        return self._worker_barrier is not None

    @property
    def distributed_mode(self):
        """Whether it is distributed training or not."""
        return bool(self._cluster_spec) and self._task_type != _TaskType.EVALUATOR

    @property
    def cluster_spec(self):
        """Returns a copy of the cluster_spec object."""
        return copy.deepcopy(self._cluster_spec)

    @property
    def task_type(self):
        """Returns the role of the corresponding task."""
        return self._task_type

    @property
    def task_id(self):
        """Returns the id or index of the corresponding task."""
        return self._task_id

    @property
    def master_target(self):
        """Returns the session master for the corresponding task to connect to."""
        return self._master_target

    @property
    def is_chief(self):
        """Returns whether the task is a chief node."""
        return self._is_chief_node

    @property
    def num_workers(self):
        """Returns number of workers in the cluster, including chief."""
        return self._num_workers

    @property
    def experimental_should_init(self):
        """Whether to run init ops."""
        return self._strategy.extended.experimental_should_init

    @property
    def should_checkpoint(self):
        """Whether to save checkpoint."""
        return self._strategy.extended.should_checkpoint

    @property
    def should_save_summary(self):
        """Whether to save summaries."""
        return self._strategy.extended.should_save_summary