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
def _verify_strategy_compatibility(self, train_distribute, eval_distribute):
    if train_distribute is not None and train_distribute.__class__ == tf.compat.v2.distribute.experimental.ParameterServerStrategy or (eval_distribute is not None and eval_distribute.__class__ == tf.compat.v2.distribute.experimental.ParameterServerStrategy):
        raise ValueError('Please use `tf.compat.v1.distribute.experimental.ParameterServerStrategy` for parameter server strategy with estimator.')