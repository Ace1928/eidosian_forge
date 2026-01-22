import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['sparse.SparseConditionalAccumulator', 'SparseConditionalAccumulator'])
class SparseConditionalAccumulator(ConditionalAccumulatorBase):
    """A conditional accumulator for aggregating sparse gradients.

  Sparse gradients are represented by `IndexedSlices`.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.

  Args:
    dtype: Datatype of the accumulated gradients.
    shape: Shape of the accumulated gradients.
    shared_name: Optional. If non-empty, this accumulator will be shared under
      the given name across multiple sessions.
    name: Optional name for the accumulator.
    reduction_type: Reduction type to use when taking the gradient.
  """

    def __init__(self, dtype, shape=None, shared_name=None, name='sparse_conditional_accumulator', reduction_type='MEAN'):
        accumulator_ref = gen_data_flow_ops.sparse_conditional_accumulator(dtype=dtype, shape=shape, shared_name=shared_name, name=name, reduction_type=reduction_type)
        super(SparseConditionalAccumulator, self).__init__(dtype, shape, accumulator_ref)

    def apply_indexed_slices_grad(self, grad, local_step=0, name=None):
        """Attempts to apply a gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.

    Args:
      grad: The gradient `IndexedSlices` to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    """
        return self.apply_grad(grad_indices=grad.indices, grad_values=grad.values, grad_shape=grad.dense_shape, local_step=local_step, name=name)

    def apply_grad(self, grad_indices, grad_values, grad_shape=None, local_step=0, name=None):
        """Attempts to apply a sparse gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.

    A sparse gradient is represented by its indices, values and possibly empty
    or None shape. Indices must be a vector representing the locations of
    non-zero entries in the tensor. Values are the non-zero slices of the
    gradient, and must have the same first dimension as indices, i.e., the nnz
    represented by indices and values must be consistent. Shape, if not empty or
    None, must be consistent with the accumulator's shape (if also provided).

    Example:
      A tensor [[0, 0], [0, 1], [2, 3]] can be represented
        indices: [1,2]
        values: [[0,1],[2,3]]
        shape: [3, 2]

    Args:
      grad_indices: Indices of the sparse gradient to be applied.
      grad_values: Values of the sparse gradient to be applied.
      grad_shape: Shape of the sparse gradient to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    """
        local_step = math_ops.cast(ops.convert_to_tensor(local_step), _dtypes.int64)
        return gen_data_flow_ops.sparse_accumulator_apply_gradient(self._accumulator_ref, local_step=local_step, gradient_indices=math_ops.cast(grad_indices, _dtypes.int64), gradient_values=grad_values, gradient_shape=math_ops.cast([] if grad_shape is None else grad_shape, _dtypes.int64), has_known_shape=grad_shape is not None, name=name)

    def take_grad(self, num_required, name=None):
        """Attempts to extract the average gradient from the accumulator.

    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.

    Once successful, the following actions are also triggered:
    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.

    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation

    Returns:
      A tuple of indices, values, and shape representing the average gradient.

    Raises:
      InvalidArgumentError: If `num_required` < 1
    """
        return gen_data_flow_ops.sparse_accumulator_take_gradient(self._accumulator_ref, num_required, dtype=self._dtype, name=name)

    def take_indexed_slices_grad(self, num_required, name=None):
        """Attempts to extract the average gradient from the accumulator.

    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.

    Once successful, the following actions are also triggered:
    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.

    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation

    Returns:
      An `IndexedSlices` holding the value of the average gradient.

    Raises:
      InvalidArgumentError: If `num_required` < 1
    """
        return_val = gen_data_flow_ops.sparse_accumulator_take_gradient(self._accumulator_ref, num_required, dtype=self._dtype, name=name)
        return indexed_slices.IndexedSlices(indices=return_val.indices, values=return_val.values, dense_shape=return_val.shape)

    def num_accumulated(self, name=None):
        """Number of gradients that have currently been aggregated in accumulator.

    Args:
      name: Optional name for the operation.

    Returns:
      Number of accumulated gradients currently in accumulator.
    """
        if name is None:
            name = '%s_NumAccumulated' % self._name
        return gen_data_flow_ops.accumulator_num_accumulated(self._accumulator_ref, name=name)

    def set_global_step(self, new_global_step, name=None):
        """Sets the global time step of the accumulator.

    The operation logs a warning if we attempt to set to a time step that is
    lower than the accumulator's own time step.

    Args:
      new_global_step: Value of new time step. Can be a variable or a constant
      name: Optional name for the operation.

    Returns:
      Operation that sets the accumulator's time step.
    """
        return gen_data_flow_ops.accumulator_set_global_step(self._accumulator_ref, math_ops.cast(ops.convert_to_tensor(new_global_step), _dtypes.int64), name=name)