from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _clip(params, ids, max_norm):
    """Helper function for _embedding_lookup_and_transform.

  This function optionally clips embeddings to an l2-norm of max_norm.

  Args:
    params: A `Tensor` of embeddings retrieved by `gather`.
    ids: The `ids` argument that was passed to `gather`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.

  Returns:
    A `Tensor` with the same type as `params`.
  """

    def _rank(x):
        """Helper function to retrieve the rank of a tensor.

    Args:
      x: Something convertible to `Tensor`.

    Returns:
      Either a pair `(rank, True)` where `rank` is an integer or a pair
      `(rank, False)` where `rank` is an integer `Tensor`. In either case,
      `rank` is the rank of `x`.
    """
        rank = ops.convert_to_tensor(x).get_shape().ndims
        if rank:
            return (rank, True)
        else:
            return (array_ops.rank(x), False)
    if max_norm is None:
        return params
    ids_rank, ids_static = _rank(ids)
    params_rank, params_static = _rank(params)
    return clip_ops.clip_by_norm(params, max_norm, axes=list(range(ids_rank, params_rank)) if ids_static and params_static else math_ops.range(ids_rank, params_rank))