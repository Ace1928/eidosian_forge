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
def _merge_dims(value, outer_axis, inner_axis):
    """Merges `outer_axis...inner_axis` of `value` into a single dimension."""
    assert outer_axis < inner_axis
    if isinstance(value, (tensor.Tensor, ragged_tensor.RaggedTensor)):
        return ragged_tensor.merge_dims(value, outer_axis, inner_axis)
    else:
        assert isinstance(value, StructuredTensor)
        fields = dict(((k, _merge_dims(v, outer_axis, inner_axis)) for k, v in value._fields.items()))
        ragged_shape = value._ragged_shape._merge_dims(outer_axis, inner_axis)
        return StructuredTensor(fields, ragged_shape)