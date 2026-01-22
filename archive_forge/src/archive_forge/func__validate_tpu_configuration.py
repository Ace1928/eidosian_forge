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