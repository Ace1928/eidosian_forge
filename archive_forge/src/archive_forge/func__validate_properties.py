from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _validate_properties(run_config):
    """Validates the properties."""

    def _validate(property_name, cond, message):
        property_value = getattr(run_config, property_name)
        if property_value is not None and (not cond(property_value)):
            raise ValueError(message)

    def _validate_delay(delay):
        """Check that delay is an integer value.

    Since this has to work for both Python2 and Python3 and PEP237 defines long
    to be basically int, we cannot just use a lambda function.
    """
        try:
            return isinstance(delay, (int, long))
        except NameError:
            return isinstance(delay, int)
    _validate('model_dir', lambda dir: dir, message='model_dir should be non-empty')
    _validate('save_summary_steps', lambda steps: steps >= 0, message='save_summary_steps should be >= 0')
    _validate('save_checkpoints_steps', lambda steps: steps >= 0, message='save_checkpoints_steps should be >= 0')
    _validate('save_checkpoints_secs', lambda secs: secs >= 0, message='save_checkpoints_secs should be >= 0')
    _validate('session_config', lambda sc: isinstance(sc, tf.compat.v1.ConfigProto), message='session_config must be instance of ConfigProto')
    _validate('keep_checkpoint_max', lambda keep_max: keep_max >= 0, message='keep_checkpoint_max should be >= 0')
    _validate('keep_checkpoint_every_n_hours', lambda keep_hours: keep_hours > 0, message='keep_checkpoint_every_n_hours should be > 0')
    _validate('log_step_count_steps', lambda num_steps: num_steps > 0, message='log_step_count_steps should be > 0')
    _validate('tf_random_seed', lambda seed: isinstance(seed, six.integer_types), message='tf_random_seed must be integer.')
    _validate('experimental_max_worker_delay_secs', _validate_delay, message='experimental_max_worker_delay_secs must be an integer if set.')
    _validate('session_creation_timeout_secs', lambda timeout_secs: timeout_secs > 0, message='session_creation_timeout_secs should be > 0')
    _validate('device_fn', lambda device_fn: six.callable(device_fn) and set(function_utils.fn_args(device_fn)) == _VALID_DEVICE_FN_ARGS, message='device_fn must be callable with exactly one argument "op".')
    _validate('protocol', lambda protocol: protocol in (None, 'grpc', 'grpc+verbs'), message='protocol should be grpc or grpc+verbs')