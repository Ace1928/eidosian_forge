from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _cast_indexed_slice_indices(a, b):
    """Cast IndexedSlice.indices from int32 to int64 where necessary.

  If `a` and `b` are both IndexedSlices, and their indices have different
  dtypes, then cast both their dtypes to `int64` (modifies `a` and `b`
  in-place).  Otherwise, does nothing.

  Args:
    a: A value, which may be an IndexedSlices.
    b: A value, which may be an IndexedSlices.
  """
    if isinstance(a, indexed_slices.IndexedSlices) and isinstance(b, indexed_slices.IndexedSlices) and (a.indices.dtype != b.indices.dtype):
        a._indices = math_ops.cast(a.indices, dtypes.int64)
        b._indices = math_ops.cast(b.indices, dtypes.int64)