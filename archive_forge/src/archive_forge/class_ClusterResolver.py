import abc
import collections
import six
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.cluster_resolver.ClusterResolver')
@six.add_metaclass(abc.ABCMeta)
class ClusterResolver(object):
    """Abstract class for all implementations of ClusterResolvers.

  This defines the skeleton for all implementations of ClusterResolvers.
  ClusterResolvers are a way for TensorFlow to communicate with various cluster
  management systems (e.g. GCE, AWS, etc...) and gives TensorFlow necessary
  information to set up distributed training.

  By letting TensorFlow communicate with these systems, we will be able to
  automatically discover and resolve IP addresses for various TensorFlow
  workers. This will eventually allow us to automatically recover from
  underlying machine failures and scale TensorFlow worker clusters up and down.

  Note to Implementors of `tf.distribute.cluster_resolver.ClusterResolver`
  subclass: In addition to these abstract methods, when task_type, task_id, and
  rpc_layer attributes are applicable, you should also implement them either as
  properties with getters or setters, or directly set the attributes
  `self._task_type`, `self._task_id`, or `self._rpc_layer` so the base class'
  getters and setters are used. See
  `tf.distribute.cluster_resolver.SimpleClusterResolver.__init__` for an
  example.

  In general, multi-client tf.distribute strategies such as
  `tf.distribute.experimental.MultiWorkerMirroredStrategy` require task_type and
  task_id properties to be available in the `ClusterResolver` they are using. On
  the other hand, these concepts are not applicable in single-client strategies,
  such as `tf.distribute.experimental.TPUStrategy`, because the program is only
  expected to be run on one task, so there should not be a need to have code
  branches according to task type and task id.

  - task_type is the name of the server's current named job (e.g. 'worker',
     'ps' in a distributed parameterized training job).
  - task_id is the ordinal index of the server within the task type.
  - rpc_layer is the protocol used by TensorFlow to communicate with other
      TensorFlow servers in a distributed environment.
  """

    @abc.abstractmethod
    def cluster_spec(self):
        """Retrieve the current state of the cluster and return a `tf.train.ClusterSpec`.

    Returns:
      A `tf.train.ClusterSpec` representing the state of the cluster at the
      moment this function is called.

    Implementors of this function must take care in ensuring that the
    ClusterSpec returned is up-to-date at the time of calling this function.
    This usually means retrieving the information from the underlying cluster
    management system every time this function is invoked and reconstructing
    a cluster_spec, rather than attempting to cache anything.
    """
        raise NotImplementedError()

    @abc.abstractmethod
    def master(self, task_type=None, task_id=None, rpc_layer=None):
        """Retrieves the name or URL of the session master.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.

    Implementors of this function must take care in ensuring that the master
    returned is up-to-date at the time to calling this function. This usually
    means retrieving the master every time this function is invoked.
    """
        raise NotImplementedError()

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        """Returns the number of accelerator cores per worker.

    This returns the number of accelerator cores (such as GPUs and TPUs)
    available per worker.

    Optionally, we allow callers to specify the task_type, and task_id, for
    if they want to target a specific TensorFlow task to query
    the number of accelerators. This is to support heterogenous environments,
    where the number of accelerators cores per host is different.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the machine we
        want to query.
      task_id: (Optional) The index of the TensorFlow task of the machine we
        want to query.
      config_proto: (Optional) Configuration for starting a new session to
        query how many accelerator cores it has.

    Returns:
      A map of accelerator types to number of cores.
    """
        master = self.master(task_type, task_id)
        devices = get_accelerator_devices(master, config_proto)
        mapping = collections.defaultdict(int)
        for device in devices:
            if task_type is not None and task_id is not None:
                job_path = '/job:%s' % task_type
                task_path = '/task:%s' % task_id
                if job_path not in device.name or task_path not in device.name:
                    continue
            mapping[device.device_type] += 1
        return mapping

    @property
    def environment(self):
        """Returns the current environment which TensorFlow is running in.

    There are two possible return values, "google" (when TensorFlow is running
    in a Google-internal environment) or an empty string (when TensorFlow is
    running elsewhere).

    If you are implementing a ClusterResolver that works in both the Google
    environment and the open-source world (for instance, a TPU ClusterResolver
    or similar), you will have to return the appropriate string depending on the
    environment, which you will have to detect.

    Otherwise, if you are implementing a ClusterResolver that will only work
    in open-source TensorFlow, you do not need to implement this property.
    """
        return ''

    @property
    def task_type(self):
        """Returns the task type this `ClusterResolver` indicates.

    In TensorFlow distributed environment, each job may have an applicable
    task type. Valid task types in TensorFlow include
    'chief': a worker that is designated with more responsibility,
    'worker': a regular worker for training/evaluation,
    'ps': a parameter server, or
    'evaluator': an evaluator that evaluates the checkpoints for metrics.

    See [Multi-worker configuration](
    https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#multi-worker_configuration)
    for more information about 'chief' and 'worker' task type, which are most
    commonly used.

    Having access to such information is useful when user needs to run specific
    code according to task types. For example,

    ```python
    cluster_spec = tf.train.ClusterSpec({
        "ps": ["localhost:2222", "localhost:2223"],
        "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
    })

    # SimpleClusterResolver is used here for illustration; other cluster
    # resolvers may be used for other source of task type/id.
    simple_resolver = SimpleClusterResolver(cluster_spec, task_type="worker",
                                            task_id=1)

    ...

    if cluster_resolver.task_type == 'worker':
      # Perform something that's only applicable on workers. This block
      # will run on this particular instance since we've specified this task to
      # be a worker in above cluster resolver.
    elif cluster_resolver.task_type == 'ps':
      # Perform something that's only applicable on parameter servers. This
      # block will not run on this particular instance.
    ```

    Returns `None` if such information is not available or is not applicable
    in the current distributed environment, such as training with
    `tf.distribute.experimental.TPUStrategy`.

    For more information, please see
    `tf.distribute.cluster_resolver.ClusterResolver`'s class doc.
    """
        return getattr(self, '_task_type', None)

    @property
    def task_id(self):
        """Returns the task id this `ClusterResolver` indicates.

    In TensorFlow distributed environment, each job may have an applicable
    task id, which is the index of the instance within its task type. This is
    useful when user needs to run specific code according to task index. For
    example,

    ```python
    cluster_spec = tf.train.ClusterSpec({
        "ps": ["localhost:2222", "localhost:2223"],
        "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
    })

    # SimpleClusterResolver is used here for illustration; other cluster
    # resolvers may be used for other source of task type/id.
    simple_resolver = SimpleClusterResolver(cluster_spec, task_type="worker",
                                            task_id=0)

    ...

    if cluster_resolver.task_type == 'worker' and cluster_resolver.task_id == 0:
      # Perform something that's only applicable on 'worker' type, id 0. This
      # block will run on this particular instance since we've specified this
      # task to be a 'worker', id 0 in above cluster resolver.
    else:
      # Perform something that's only applicable on other ids. This block will
      # not run on this particular instance.
    ```

    Returns `None` if such information is not available or is not applicable
    in the current distributed environment, such as training with
    `tf.distribute.cluster_resolver.TPUClusterResolver`.

    For more information, please see
    `tf.distribute.cluster_resolver.ClusterResolver`'s class docstring.
    """
        return getattr(self, '_task_id', None)

    @task_type.setter
    def task_type(self, task_type):
        """Setter of `task_type` property. See `task_type` property doc."""
        self._task_type = task_type

    @task_id.setter
    def task_id(self, task_id):
        """Setter of `task_id` property. See `task_type` property doc."""
        self._task_id = task_id