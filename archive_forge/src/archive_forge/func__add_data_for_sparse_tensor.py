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
def _add_data_for_sparse_tensor(self, tensor, weight, indices, values, weights, int_zeros, float_zeros, path, feature):
    sample_indices = math_ops.cast(tensor.indices, dtypes.int32)
    if tensor.shape.rank == 2:
        if not feature.output_shape and feature.max_sequence_length > 0:
            sample_indices = array_ops.pad(sample_indices, paddings=[[0, 0], [0, 1]])
    elif feature.max_sequence_length > 0:
        logging.warning('Input tensor is rank %d which is above 2, the max_sequence_length setting will be ignored.', tensor.shape.rank)
    indices.append(sample_indices)
    values.append(math_ops.cast(tensor.values, dtypes.int64))
    if weight is not None:
        if not isinstance(weight, sparse_tensor.SparseTensor):
            raise ValueError('Weight for {} is type {} which does not match type input which is SparseTensor.'.format(path, type(weight)))
        weights.append(math_ops.cast(weight.values, dtypes.float32))
    else:
        weights.append(float_zeros)