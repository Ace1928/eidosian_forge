import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _dynamic_ragged_shape_from_tensor(field, dtype=None) -> dynamic_ragged_shape.DynamicRaggedShape:
    """Extension of DynamicRaggedShape.from_tensor to support StructuredTensor."""
    if isinstance(field, StructuredTensor):
        return field._ragged_shape
    shape = array_ops.shape_v2(field, out_type=dtype)
    if isinstance(shape, tensor.Tensor):
        return dynamic_ragged_shape.DynamicRaggedShape(row_partitions=[], inner_shape=shape)
    elif isinstance(shape, dynamic_ragged_shape.DynamicRaggedShape):
        return shape
    raise TypeError(f'Expected shape tf.shape({field}) to return a Tensor or a DynamicRaggedShape. Instead, got: {shape}.')