import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _raise_error_for_non_direct_inputs(self, features):
    """Checks all tensors in features to see if they are a direct input."""
    for path, input_tensor in nest.flatten_with_joined_string_paths(features, expand_composites=True):
        if input_tensor.op.type == 'Placeholder':
            continue
        try:
            is_input = input_tensor.op.get_attr('_tpu_input_identity')
        except ValueError:
            is_input = False
        if not is_input:
            raise ValueError('Received input tensor {} which is the output of op {} (type {}) which does not have the `_tpu_input_identity` attr. Please ensure that the inputs to this layer are taken directly from the arguments of the function called by strategy.run. Two possible causes are: dynamic batch size support or you are using a keras layer and are not passing tensors which match the dtype of the `tf.keras.Input`s.If you are triggering dynamic batch size support, you can disable it by passing tf.distribute.RunOptions(experimental_enable_dynamic_batch_size=False) to the options argument of strategy.run().'.format(path, input_tensor.op.name, input_tensor.op.type))