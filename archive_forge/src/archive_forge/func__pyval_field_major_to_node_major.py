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
def _pyval_field_major_to_node_major(keys, values, depth):
    """Regroup each field (k, v) from dict-of-list to list-of-dict.

  Given a "field-major" encoding of the StructuredTensor (which maps each key to
  a single nested list containing the values for all structs), return a
  corresponding "node-major" encoding, consisting of a nested list of dicts.

  Args:
    keys: The field names (list of string).  Must not be empty.
    values: The field values (list of python values).  Must have the same length
      as `keys`.
    depth: The list depth at which dictionaries should be created.

  Returns:
    A nested list of dict, with depth `depth`.
  """
    assert keys
    if depth == 0:
        return dict(zip(keys, values))
    nvals = len(values[0])
    assert all((nvals == len(values[i]) for i in range(1, len(values))))
    return [_pyval_field_major_to_node_major(keys, value_slice, depth - 1) for value_slice in zip(*values)]