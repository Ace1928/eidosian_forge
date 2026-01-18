from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops.gen_state_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
Generalization of `tf.compat.v1.scatter_update` to axis different than 0.

  Analogous to `batch_gather`. This assumes that `ref`, `indices` and `updates`
  have a series of leading dimensions that are the same for all of them, and the
  updates are performed on the last dimension of indices. In other words, the
  dimensions should be the following:

  `num_prefix_dims = indices.ndims - 1`
  `batch_dim = num_prefix_dims + 1`
  `updates.shape = indices.shape + var.shape[batch_dim:]`

  where

  `updates.shape[:num_prefix_dims]`
  `== indices.shape[:num_prefix_dims]`
  `== var.shape[:num_prefix_dims]`

  And the operation performed can be expressed as:

  `var[i_1, ..., i_n, indices[i_1, ..., i_n, j]] = updates[i_1, ..., i_n, j]`

  When indices is a 1D tensor, this operation is equivalent to
  `tf.compat.v1.scatter_update`.

  To avoid this operation there would be 2 alternatives:
  1) Reshaping the variable by merging the first `ndims` dimensions. However,
     this is not possible because `tf.reshape` returns a Tensor, which we
     cannot use `tf.compat.v1.scatter_update` on.
  2) Looping over the first `ndims` of the variable and using
     `tf.compat.v1.scatter_update` on the subtensors that result of slicing the
     first
     dimension. This is a valid option for `ndims = 1`, but less efficient than
     this implementation.

  See also `tf.compat.v1.scatter_update` and `tf.compat.v1.scatter_nd_update`.

  Args:
    ref: `Variable` to scatter onto.
    indices: Tensor containing indices as described above.
    updates: Tensor of updates to apply to `ref`.
    use_locking: Boolean indicating whether to lock the writing operation.
    name: Optional scope name string.

  Returns:
    Ref to `variable` after it has been modified.

  Raises:
    ValueError: If the initial `ndims` of `ref`, `indices`, and `updates` are
        not the same.
  