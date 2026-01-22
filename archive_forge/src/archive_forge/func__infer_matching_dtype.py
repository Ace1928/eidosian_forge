import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _infer_matching_dtype(tensors, dtype_hierarchy):
    """Infers a matching dtype for tensors, and casts them to that dtype."""
    assert all((t.dtype in dtype_hierarchy for t in tensors))
    inferred_dtype = max([t.dtype for t in tensors], key=dtype_hierarchy.index)
    return [math_ops.cast(t, inferred_dtype) for t in tensors]