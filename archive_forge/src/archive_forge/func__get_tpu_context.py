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
def _get_tpu_context(config, train_batch_size, eval_batch_size, predict_batch_size, use_tpu, eval_on_tpu, embedding_config_spec):
    """Returns an instance of `_InternalTPUContext`."""
    if config.tpu_config.num_shards == 1 and config.tpu_config.num_cores_per_replica is None:
        if embedding_config_spec is not None:
            raise ValueError('Setting TPUConfig.num_shards==1 is unsupported when embedding_config_spec is not None.')
        tf.compat.v1.logging.warn('Setting TPUConfig.num_shards==1 is an unsupported behavior. Please fix as soon as possible (leaving num_shards as None.)')
        return _OneCoreTPUContext(config, train_batch_size, eval_batch_size, predict_batch_size, use_tpu)
    return _InternalTPUContext(config, train_batch_size, eval_batch_size, predict_batch_size, use_tpu, eval_on_tpu, embedding_config_spec)