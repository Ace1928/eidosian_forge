from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _rnn_step(time, sequence_length, min_sequence_length, max_sequence_length, zero_output, state, call_cell, state_size, skip_conditionals=False):
    """Calculate one step of a dynamic RNN minibatch.

  Returns an (output, state) pair conditioned on `sequence_length`.
  When skip_conditionals=False, the pseudocode is something like:

  if t >= max_sequence_length:
    return (zero_output, state)
  if t < min_sequence_length:
    return call_cell()

  # Selectively output zeros or output, old state or new state depending
  # on whether we've finished calculating each row.
  new_output, new_state = call_cell()
  final_output = np.vstack([
    zero_output if time >= sequence_length[r] else new_output_r
    for r, new_output_r in enumerate(new_output)
  ])
  final_state = np.vstack([
    state[r] if time >= sequence_length[r] else new_state_r
    for r, new_state_r in enumerate(new_state)
  ])
  return (final_output, final_state)

  Args:
    time: int32 `Tensor` scalar.
    sequence_length: int32 `Tensor` vector of size [batch_size].
    min_sequence_length: int32 `Tensor` scalar, min of sequence_length.
    max_sequence_length: int32 `Tensor` scalar, max of sequence_length.
    zero_output: `Tensor` vector of shape [output_size].
    state: Either a single `Tensor` matrix of shape `[batch_size, state_size]`,
      or a list/tuple of such tensors.
    call_cell: lambda returning tuple of (new_output, new_state) where
      new_output is a `Tensor` matrix of shape `[batch_size, output_size]`.
      new_state is a `Tensor` matrix of shape `[batch_size, state_size]`.
    state_size: The `cell.state_size` associated with the state.
    skip_conditionals: Python bool, whether to skip using the conditional
      calculations.  This is useful for `dynamic_rnn`, where the input tensor
      matches `max_sequence_length`, and using conditionals just slows
      everything down.

  Returns:
    A tuple of (`final_output`, `final_state`) as given by the pseudocode above:
      final_output is a `Tensor` matrix of shape [batch_size, output_size]
      final_state is either a single `Tensor` matrix, or a tuple of such
        matrices (matching length and shapes of input `state`).

  Raises:
    ValueError: If the cell returns a state tuple whose length does not match
      that returned by `state_size`.
  """
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)
    copy_cond = time >= sequence_length

    def _copy_one_through(output, new_output):
        if isinstance(output, tensor_array_ops.TensorArray):
            return new_output
        if output.shape.rank == 0:
            return new_output
        with ops.colocate_with(new_output):
            return array_ops.where(copy_cond, output, new_output)

    def _copy_some_through(flat_new_output, flat_new_state):
        flat_new_output = [_copy_one_through(zero_output, new_output) for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [_copy_one_through(state, new_state) for state, new_state in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output, new_state = call_cell()
        nest.assert_same_structure(zero_output, new_output)
        nest.assert_same_structure(state, new_state)
        flat_new_state = nest.flatten(new_state)
        flat_new_output = nest.flatten(new_output)
        return cond.cond(time < min_sequence_length, lambda: flat_new_output + flat_new_state, lambda: _copy_some_through(flat_new_output, flat_new_state))
    if skip_conditionals:
        new_output, new_state = call_cell()
        nest.assert_same_structure(zero_output, new_output)
        nest.assert_same_structure(state, new_state)
        new_state = nest.flatten(new_state)
        new_output = nest.flatten(new_output)
        final_output_and_state = _copy_some_through(new_output, new_state)
    else:
        empty_update = lambda: flat_zero_output + flat_state
        final_output_and_state = cond.cond(time >= max_sequence_length, empty_update, _maybe_copy_some_through)
    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError(f'Internal error: state and output were not concatenated correctly. Received state length: {len(flat_state)}, output length: {len(flat_zero_output)}. Expected contatenated length: {len(final_output_and_state)}.')
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]
    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        if not isinstance(substate, tensor_array_ops.TensorArray):
            substate.set_shape(flat_substate.get_shape())
    final_output = nest.pack_sequence_as(structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(structure=state, flat_sequence=final_state)
    return (final_output, final_state)