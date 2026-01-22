import collections
import re
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export
def _add_ragged_key(self, key, value_type, split_type):
    """Adds a ragged key & dtype, checking for duplicates."""
    if key in self.ragged_keys:
        original_value_type = self.ragged_value_types[self.ragged_keys.index(key)]
        original_split_type = self.ragged_split_types[self.ragged_keys.index(key)]
        if original_value_type != value_type:
            raise ValueError(f'Conflicting type {original_value_type} vs {value_type} for feature {key}.')
        if original_split_type != split_type:
            raise ValueError(f'Conflicting partition type {original_split_type} vs {split_type} for feature {key}.')
    else:
        self.ragged_keys.append(key)
        self.ragged_value_types.append(value_type)
        self.ragged_split_types.append(split_type)