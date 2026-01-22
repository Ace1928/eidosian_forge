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
def _matmul_3d_with_map_fn(a, b, **kwargs):
    """Multiplies batches of 2D matrices using map_fn.

  `output[n, i, k]` = sum_j (a[n, i, j] * b[n, j, k])` (for all `n`, `i`, `k`).

  Requires that `a[n, i].nrows()` == `b[n].nrows()` (for all `n` and `i`).

  Args:
    a: A 3D Tensor or RaggedTensor with `shape=[B, I, J]`, where dimensions `I`
      and `J` may be ragged.
    b: A 3D Tensor or RaggedTensor with `shape=[B, J, K]`, where dimensions `J`
      and `K` may be ragged.
    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).

  Returns:
    A 3D RaggedTensor with `shape=[B, (I), (K)]`.
  """
    if isinstance(b, ragged_tensor.RaggedTensor) and (b.ragged_rank == 2 or kwargs.get('transpose_b') or kwargs.get('adjoint_b')):
        output_ragged_rank = 2
    else:
        output_ragged_rank = 1

    def single_batch_matmul(x):
        out = _matmul_2d(x[0], x[1], **kwargs)
        if output_ragged_rank == 2:
            out = ragged_tensor.RaggedTensor.from_tensor(out)
        return out
    fn_out_shape = None
    row_splits_dtype = a.row_splits.dtype if isinstance(a, ragged_tensor.RaggedTensor) else b.row_splits.dtype
    output_type = kwargs['output_type']
    if output_type is None:
        output_type = a.dtype
    spec = ragged_tensor.RaggedTensorSpec(shape=fn_out_shape, dtype=output_type, ragged_rank=output_ragged_rank - 1, row_splits_dtype=row_splits_dtype)
    result = map_fn.map_fn(single_batch_matmul, elems=(a, b), fn_output_signature=spec)
    if kwargs.get('transpose_a') or kwargs.get('adjoint_a'):
        result._set_shape(a.shape[:-2] + a.shape[-1:] + [None])
    else:
        result._set_shape(a.shape[:-2] + a.shape[-2:-1] + [None])
    if kwargs.get('transpose_b') or kwargs.get('adjoint_b'):
        result._set_shape(b.shape[:-2] + [None] + b.shape[-2:-1])
    else:
        result._set_shape(b.shape[:-2] + [None] + b.shape[-1:])
    return result