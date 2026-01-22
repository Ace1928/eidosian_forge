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
def _add_sparse_key(self, key, dtype):
    """Adds a sparse key & dtype, checking for duplicates."""
    if key in self.sparse_keys:
        original_dtype = self.sparse_types[self.sparse_keys.index(key)]
        if original_dtype != dtype:
            raise ValueError(f'Conflicting type {original_dtype} vs {dtype} for feature {key}.')
    else:
        self.sparse_keys.append(key)
        self.sparse_types.append(dtype)