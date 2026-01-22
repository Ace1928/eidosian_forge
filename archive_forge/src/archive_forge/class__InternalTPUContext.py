from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
class _InternalTPUContext(object):
    """A context holds immutable states of TPU computation.

  This immutable object holds TPUEstimator config, train/eval batch size, and
  `TPUEstimator.use_tpu`, which is expected to be passed around. It also
  provides utility functions, based on the current state, to determine other
  information commonly required by TPU computation, such as TPU device names,
  TPU hosts, shard batch size, etc.

  if eval_on_tpu is False, then execution of eval on TPU is disabled.
  if eval_on_tpu is True, but use_tpu is False, a warning is issued,
  and TPU execution is disabled for all modes.

  N.B. As `mode` is not immutable state in Estimator, but essential to
  distinguish between TPU training and evaluation, a common usage for
  _InternalTPUContext with `mode` is as follows:
  ```
  with _ctx.with_mode(mode) as ctx:
    if ctx.is_running_on_cpu():
       ...
  ```
  """

    def __init__(self, config, train_batch_size, eval_batch_size, predict_batch_size, use_tpu, eval_on_tpu=True, embedding_config_spec=None):
        self._config = config
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._predict_batch_size = predict_batch_size
        self._use_tpu = use_tpu
        tf.compat.v1.logging.info('_TPUContext: eval_on_tpu %s', eval_on_tpu)
        if not use_tpu and eval_on_tpu:
            tf.compat.v1.logging.warn('eval_on_tpu ignored because use_tpu is False.')
        self._eval_on_tpu = eval_on_tpu
        self._model_parallelism_enabled = use_tpu and config.tpu_config.num_cores_per_replica
        self._mode = None
        num_cores_per_replica = config.tpu_config.num_cores_per_replica
        if self._model_parallelism_enabled:
            self._computation_shape = _NUM_CORES_TO_COMPUTATION_SHAPE[num_cores_per_replica]
        else:
            self._computation_shape = None
        self._lazy_tpu_system_metadata_dict = {}
        self._lazy_device_assignment_dict = {}
        self._lazy_validation_dict = {}
        self._embedding_config_spec = embedding_config_spec
        self._lazy_embedding_config_dict = {}

    def _assert_mode(self):
        if self._mode is None:
            raise RuntimeError('`mode` needs to be set via contextmanager `with_mode`.')
        return self._mode

    @contextmanager
    def with_mode(self, mode):
        new_ctx = copy.copy(self)
        new_ctx._mode = mode
        yield new_ctx

    @property
    def mode(self):
        return self._assert_mode()

    def _get_master_address(self):
        mode = self._assert_mode()
        config = self._config
        master = config.master if mode != model_fn_lib.ModeKeys.EVAL else config.evaluation_master
        return master

    def _get_tpu_system_metadata(self):
        """Gets the (maybe cached) TPU system metadata."""
        master = self._get_master_address()
        tpu_system_metadata = self._lazy_tpu_system_metadata_dict.get(master)
        if tpu_system_metadata is not None:
            return tpu_system_metadata
        cluster_def = None
        if self._config.session_config and self._config.session_config.cluster_def.job:
            cluster_def = self._config.session_config.cluster_def
        tpu_system_metadata = tpu_system_metadata_lib._query_tpu_system_metadata(master, cluster_def=cluster_def, query_topology=self.model_parallelism_enabled)
        self._lazy_tpu_system_metadata_dict[master] = tpu_system_metadata
        return tpu_system_metadata

    def _get_device_assignment(self):
        """Gets the (maybe cached) TPU device assignment."""
        master = self._get_master_address()
        device_assignment = self._lazy_device_assignment_dict.get(master)
        if device_assignment is not None:
            return device_assignment
        tpu_system_metadata = self._get_tpu_system_metadata()
        device_assignment = tpu_device_assignment.device_assignment(tpu_system_metadata.topology, computation_shape=self._computation_shape, num_replicas=self.num_replicas)
        tf.compat.v1.logging.info('num_cores_per_replica: %s', str(self._config.tpu_config.num_cores_per_replica))
        tf.compat.v1.logging.info('computation_shape: %s', str(self._computation_shape))
        tf.compat.v1.logging.info('num_replicas: %d', self.num_replicas)
        tf.compat.v1.logging.info('device_assignment.topology.device_coordinates: %s', str(device_assignment.topology.device_coordinates))
        tf.compat.v1.logging.info('device_assignment.core_assignment: %s', str(device_assignment.core_assignment))
        self._lazy_device_assignment_dict[master] = device_assignment
        return device_assignment

    @property
    def tensor_core_embedding_columns(self):
        if self._embedding_config_spec:
            return self._embedding_config_spec.tensor_core_feature_columns
        return None

    @property
    def embedding_config(self):
        """Returns the embedding config based on current mode."""
        master = self._get_master_address()
        if master in self._lazy_embedding_config_dict:
            embedding_config = self._lazy_embedding_config_dict[master]
        else:
            embedding_config = None
            if self._use_tpu and self._embedding_config_spec:
                embedding_config = _tpu_estimator_embedding.EmbeddingConfig(self._embedding_config_spec, self._train_batch_size, self._eval_batch_size, self.num_hosts, self.num_cores, self.config)
                if not embedding_config.has_embedding_tables():
                    embedding_config = None
            self._lazy_embedding_config_dict[master] = embedding_config
        if embedding_config is not None:
            mode = self._assert_mode()
            embedding_config.tpu_embedding = embedding_config.get_tpu_embedding(mode)
        return embedding_config

    @property
    def allow_per_host_v2_parallel_get_next(self):
        return self._config.tpu_config.experimental_allow_per_host_v2_parallel_get_next

    @property
    def feed_hook(self):
        return self._config.tpu_config.experimental_feed_hook

    @property
    def model_parallelism_enabled(self):
        return self._model_parallelism_enabled

    @property
    def input_partition_dims(self):
        return self._config.tpu_config.input_partition_dims

    @property
    def device_assignment(self):
        return self._get_device_assignment() if self._model_parallelism_enabled else None

    @property
    def num_of_cores_per_host(self):
        metadata = self._get_tpu_system_metadata()
        return metadata.num_of_cores_per_host

    @property
    def num_cores(self):
        metadata = self._get_tpu_system_metadata()
        return metadata.num_cores

    @property
    def num_of_replicas_per_host(self):
        """Return the number of replicas per host."""
        if self.model_parallelism_enabled:
            return self.num_replicas // self.num_hosts
        else:
            return self.num_of_cores_per_host

    @property
    def num_replicas(self):
        """Compute the total number of replicas."""
        num_cores_in_system = self.num_cores
        if self.model_parallelism_enabled:
            num_cores_per_replica = self._config.tpu_config.num_cores_per_replica
            if num_cores_per_replica > num_cores_in_system:
                raise ValueError('The num of cores required by the model parallelism, specified by TPUConfig.num_cores_per_replica, is larger than the total num of TPU cores in the system. num_cores_per_replica: {}, num cores in the system: {}'.format(num_cores_per_replica, num_cores_in_system))
            if num_cores_in_system % num_cores_per_replica != 0:
                raise RuntimeError('The num of cores in the system ({}) is not divisible by the num of cores ({}) required by the model parallelism, specified by TPUConfig.num_cores_per_replica. This should never happen!'.format(num_cores_in_system, num_cores_per_replica))
            return num_cores_in_system // num_cores_per_replica
        else:
            return num_cores_in_system

    @property
    def num_hosts(self):
        metadata = self._get_tpu_system_metadata()
        return metadata.num_hosts

    @property
    def config(self):
        return self._config

    def is_input_sharded_per_core(self):
        """Return true if input_fn is invoked per-core (other than per-host)."""
        mode = self._assert_mode()
        return mode == model_fn_lib.ModeKeys.TRAIN and self._config.tpu_config.per_host_input_for_training is tpu_config.InputPipelineConfig.PER_SHARD_V1

    def is_input_per_host_with_iterators(self):
        """Return true if input_fn should be run in the per-host v2 config."""
        return self._config.tpu_config.per_host_input_for_training is tpu_config.InputPipelineConfig.PER_HOST_V2

    def is_input_broadcast_with_iterators(self):
        """Return true if input_fn should be run in the full_replicae config."""
        return self._config.tpu_config.per_host_input_for_training is tpu_config.InputPipelineConfig.BROADCAST or self.is_input_slice_broadcast_to_all_cores()

    def is_input_slice_broadcast_to_all_cores(self):
        """Return true if input_fn is invoked once and broadcast to other hosts."""
        mode = self._assert_mode()
        return mode != model_fn_lib.ModeKeys.TRAIN and self._config.tpu_config.eval_training_input_configuration is tpu_config.InputPipelineConfig.SLICED

    def is_replica_across_hosts(self):
        """Return true if single replica is across multiple hosts."""
        num_cores_per_replica = self._config.tpu_config.num_cores_per_replica
        num_cores_per_host = self._get_tpu_system_metadata().num_of_cores_per_host
        return num_cores_per_replica is not None and num_cores_per_replica > num_cores_per_host

    def is_running_on_cpu(self, is_export_mode=False):
        """Determines whether the input_fn and model_fn should be invoked on CPU.

    This API also validates user provided configuration, such as batch size,
    according the lazy initialized TPU system metadata.

    Args:
      is_export_mode: Indicates whether the current mode is for exporting the
        model, when mode == PREDICT. Only with this bool, we could tell whether
        user is calling the Estimator.predict or Estimator.export_savedmodel,
        which are running on TPU and CPU respectively. Parent class Estimator
        does not distinguish these two.

    Returns:
      bool, whether current input_fn or model_fn should be running on CPU.

    Raises:
      ValueError: any configuration is invalid.
    """
        is_running_on_cpu = self._is_running_on_cpu(is_export_mode)
        if not is_running_on_cpu:
            self._validate_tpu_configuration()
        return is_running_on_cpu

    def _is_running_on_cpu(self, is_export_mode):
        """Determines whether the input_fn and model_fn should be invoked on CPU."""
        mode = self._assert_mode()
        if not self._use_tpu:
            return True
        if mode == model_fn_lib.ModeKeys.EVAL and (not self._eval_on_tpu):
            tf.compat.v1.logging.info('_is_running_on_cpu: eval_on_tpu disabled')
            return True
        if is_export_mode:
            return True
        return False

    @property
    def global_batch_size(self):
        mode = self._assert_mode()
        if mode == model_fn_lib.ModeKeys.TRAIN:
            return self._train_batch_size
        elif mode == model_fn_lib.ModeKeys.EVAL:
            return self._eval_batch_size
        elif mode == model_fn_lib.ModeKeys.PREDICT:
            return self._predict_batch_size
        else:
            return None

    @property
    def batch_size_for_input_fn(self):
        """Returns the shard batch size for `input_fn`."""
        global_batch_size = self.global_batch_size
        if self.is_running_on_cpu() or self.is_input_broadcast_with_iterators():
            return global_batch_size
        if self.is_input_sharded_per_core() or self.is_input_per_host_with_iterators() or self.is_replica_across_hosts():
            return global_batch_size // self.num_replicas
        else:
            return global_batch_size // self.num_hosts

    @property
    def batch_size_for_model_fn(self):
        """Returns the shard batch size for `model_fn`."""
        global_batch_size = self.global_batch_size
        if self.is_running_on_cpu() or (self.is_input_broadcast_with_iterators() and (not self.is_input_slice_broadcast_to_all_cores())):
            return global_batch_size
        return global_batch_size // self.num_replicas

    @property
    def master_job(self):
        """Returns the job name to use to place TPU computations on.

    Returns:
      A string containing the job name, or None if no job should be specified.

    Raises:
      ValueError: If the user needs to specify a tpu_job_name, because we are
        unable to infer the job name automatically, or if the user-specified job
        names are inappropriate.
    """
        run_config = self._config
        if run_config.tpu_config.tpu_job_name:
            return run_config.tpu_config.tpu_job_name
        mode = self._assert_mode()
        master = run_config.evaluation_master if mode == model_fn_lib.ModeKeys.EVAL else run_config.master
        cluster_def = run_config.session_config.cluster_def if run_config.session_config else None
        try:
            master_job = tpu_system_metadata_lib.master_job(master, cluster_def)
        except ValueError as e:
            raise ValueError(str(e) + ' Please specify a tpu_job_name as part of your TPUConfig.')
        return master_job

    @property
    def tpu_host_placement_function(self):
        """Returns the TPU host place function."""
        master = self.master_job

        def _placement_function(_sentinal=None, replica_id=None, host_id=None):
            """Return the host device given replica_id or host_id."""
            assert _sentinal is None
            if replica_id is not None and host_id is not None:
                raise RuntimeError('replica_id and host_id can have only one non-None value.')
            if master is None:
                return '/replica:0/task:0/device:CPU:0'
            else:
                if replica_id is not None:
                    if self.model_parallelism_enabled:
                        return self.device_assignment.host_device(replica=replica_id, job=master)
                    else:
                        host_id = replica_id / self.num_of_cores_per_host
                return '/job:%s/task:%d/device:CPU:0' % (master, host_id)
        return _placement_function

    @property
    def tpu_device_placement_function(self):
        """Returns a TPU device placement Fn."""
        master = self.master_job
        job_device = '' if master is None else '/job:%s' % master

        def _placement_function(i):
            if self.model_parallelism_enabled:
                return self.device_assignment.tpu_device(replica=i, job=master)
            else:
                num_of_cores_per_host = self.num_of_cores_per_host
                host_id = i / num_of_cores_per_host
                ordinal_id = i % num_of_cores_per_host
                return '%s/task:%d/device:TPU:%d' % (job_device, host_id, ordinal_id)
        return _placement_function

    def tpu_ordinal_function(self, host_id):
        """Returns the TPU ordinal fn."""

        def _tpu_ordinal_function(shard_index_in_host):
            """Return the TPU ordinal associated with a shard.

      Required because the enqueue ops are placed on CPU.

      Args:
        shard_index_in_host: the shard index

      Returns:
        The ordinal of the TPU device the shard's infeed should be placed on.
      """
            if self.model_parallelism_enabled:
                replica = self.device_assignment.lookup_replicas(host_id, 0)[shard_index_in_host]
                return self.device_assignment.tpu_ordinal(replica=replica)
            else:
                return shard_index_in_host % self.num_of_cores_per_host
        return _tpu_ordinal_function

    def _validate_tpu_configuration(self):
        """Validates the configuration based on the TPU system metadata."""
        mode = self._assert_mode()
        if self._lazy_validation_dict.get(mode):
            return
        num_cores = self.num_cores
        num_replicas = self.num_replicas
        num_hosts = self.num_hosts
        if not num_cores:
            tpu_system_metadata = self._get_tpu_system_metadata()
            raise RuntimeError('Cannot find any TPU cores in the system. Please double check Tensorflow master address and TPU worker(s). Available devices are {}.'.format(tpu_system_metadata.devices))
        if self._config.tpu_config.num_shards:
            user_provided_num_replicas = self._config.tpu_config.num_shards
            if user_provided_num_replicas != num_replicas:
                message = 'TPUConfig.num_shards is not set correctly. According to TPU system metadata for Tensorflow master ({}): num_replicas should be ({}), got ({}). For non-model-parallelism, num_replicas should be the total num of TPU cores in the system. For model-parallelism, the total number of TPU cores should be num_cores_per_replica * num_replicas. Please set it accordingly or leave it as `None`'.format(self._get_master_address(), num_replicas, user_provided_num_replicas)
                raise ValueError(message)
        if self._config.tpu_config.num_cores_per_replica and (not self.is_input_per_host_with_iterators()):
            num_cores_per_replica = self._config.tpu_config.num_cores_per_replica
            num_cores_per_host = self._get_tpu_system_metadata().num_of_cores_per_host
            if num_cores_per_replica > num_cores_per_host:
                raise ValueError('Except the PER_HOST_V2 mode, the num of cores required by model parallelism specified by TPUConfig.num_cores_per_replica should be less than or equal to the num_cores_per_host. num_cores_per_replica: {}, num_cores_per_host: {}'.format(num_cores_per_replica, num_cores_per_host))
        if mode == model_fn_lib.ModeKeys.TRAIN:
            if self._train_batch_size % num_replicas != 0 and (not self.is_input_broadcast_with_iterators()):
                raise ValueError('train batch size {} must be divisible by number of replicas {}'.format(self._train_batch_size, num_replicas))
        elif mode == model_fn_lib.ModeKeys.EVAL:
            if self._eval_batch_size is None:
                raise ValueError('eval_batch_size in TPUEstimator constructor cannot be `None` if .evaluate is running on TPU.')
            if self._eval_batch_size % num_replicas != 0 and (not self.is_input_broadcast_with_iterators()):
                raise ValueError('eval batch size {} must be divisible by number of replicas {}'.format(self._eval_batch_size, num_replicas))
            if num_hosts != 1 and (not self.is_input_broadcast_with_iterators()) and (not self.is_input_per_host_with_iterators()):
                raise ValueError('TPUEstimator.evaluate is only supported under three conditions: 1. num_hosts=1; 2. BROADCAST mode; 3. PER_HOST_V2 mode. mode: {}; num_hosts: {}; num_replicas=1:{}'.format(self._config.tpu_config.per_host_input_for_training, num_hosts, num_replicas))
            if num_hosts > 1 and self.is_input_per_host_with_iterators():
                tf.compat.v1.logging.warn('Running TPUEstimator.evaluate for input mode PER_HOST_V2 and num_hosts %d', num_hosts)
        else:
            assert mode == model_fn_lib.ModeKeys.PREDICT
            if self._predict_batch_size is None:
                raise ValueError('predict_batch_size in TPUEstimator constructor cannot be `None` if .predict is running on TPU.')
            if self._predict_batch_size % num_replicas != 0 and (not self.is_input_broadcast_with_iterators()):
                raise ValueError('predict batch size {} must be divisible by number of replicas {}'.format(self._predict_batch_size, num_replicas))
            if num_hosts != 1 and (not self.is_input_broadcast_with_iterators()) and (not (num_replicas == 1 and self.is_input_per_host_with_iterators())):
                raise ValueError('TPUEstimator.predict is only supported under three conditions: 1. num_hosts=1; 2. BROADCAST mode; 3. PER_HOST_V2 mode with num_replicas=1. mode: {}; num_hosts: {}; num_replicas=1:{}'.format(self._config.tpu_config.per_host_input_for_training, num_hosts, num_replicas))
        self._lazy_validation_dict[mode] = True

    def device_for_replica(self, replica_id):
        """Returns the tuple of (CPU device and device ordinal) for replica.

    This should be used for full replicate for non-model-parallelism.

    Args:
       replica_id: Int, the replica index.

    Returns:
       A tuple of device spec for CPU device and int device ordinal.
    """
        master = self.master_job
        if self.model_parallelism_enabled:
            return (self.device_assignment.host_device(replica=replica_id, job=master), self.device_assignment.tpu_ordinal(replica=replica_id))
        job_device = '' if master is None else '/job:%s' % master
        num_of_replicas_per_host = self.num_of_replicas_per_host
        assert num_of_replicas_per_host > 0, 'Got num_of_replicas_per_host: {}'.format(num_of_replicas_per_host)
        host_id = replica_id / num_of_replicas_per_host
        ordinal_id = replica_id % num_of_replicas_per_host
        host_device = '%s/task:%d/device:CPU:0' % (job_device, host_id)
        return (host_device, ordinal_id)