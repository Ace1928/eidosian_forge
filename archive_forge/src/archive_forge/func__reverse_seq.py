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
def _reverse_seq(input_seq, lengths):
    """Reverse a list of Tensors up to specified lengths.

  Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, n_features)
      or nested tuples of tensors.
    lengths:   A `Tensor` of dimension batch_size, containing lengths for each
      sequence in the batch. If "None" is specified, simply reverses the list.

  Returns:
    time-reversed sequence
  """
    if lengths is None:
        return list(reversed(input_seq))
    flat_input_seq = tuple((nest.flatten(input_) for input_ in input_seq))
    flat_results = [[] for _ in range(len(input_seq))]
    for sequence in zip(*flat_input_seq):
        input_shape = tensor_shape.unknown_shape(rank=sequence[0].get_shape().rank)
        for input_ in sequence:
            input_shape.assert_is_compatible_with(input_.get_shape())
            input_.set_shape(input_shape)
        s_joined = array_ops_stack.stack(sequence)
        s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
        result = array_ops_stack.unstack(s_reversed)
        for r, flat_result in zip(result, flat_results):
            r.set_shape(input_shape)
            flat_result.append(r)
    results = [nest.pack_sequence_as(structure=input_, flat_sequence=flat_result) for input_, flat_result in zip(input_seq, flat_results)]
    return results