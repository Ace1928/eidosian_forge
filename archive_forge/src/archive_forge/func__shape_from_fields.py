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
def _shape_from_fields(fields, rank: int, dtype: dtypes.DType) -> Optional[dynamic_ragged_shape.DynamicRaggedShape]:
    """Given fields, rank, and dtype, create a shape."""
    field_shape = None
    for k, field in fields.items():
        try:
            next_field_shape_raw = _dynamic_ragged_shape_from_tensor(field, dtype=dtype)
            next_field_shape = next_field_shape_raw[:rank]
            field_shape = _merge_with_optional(field_shape, next_field_shape)
        except Exception as err:
            raise ValueError(f'Error in shape of {k}') from err
    return field_shape