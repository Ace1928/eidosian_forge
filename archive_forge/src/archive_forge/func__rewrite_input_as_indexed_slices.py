from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.util import nest
def _rewrite_input_as_indexed_slices(body_grad_graph, grad_output_slices, forward_input, loop_vars):
    """Rewrites grad_output_slices's corresponding input to be an IndexedSlices.

  This rewrite requires that forward_input was captured in the forward loop,
  i.e. is not a user-specified loop variable. This is important because the
  rewrite assumes that forward_input is passed through to its corresponding
  output unchanged. This assumption is used in _rewrite_input_as_indexed_slices,
  which depends on the exact gradient structure produced by the input's fanout.

  This can yield a more efficient computation than using
  _rewrite_output_as_tensor, since it preserves the IndexedSlices structure
  instead of converting the IndexedSlices to a dense Tensor.

  Args:
    body_grad_graph: _WhileBodyGradFuncGraph.
    grad_output_slices: IndexedSlices output of body_grad_graph.
    forward_input: the corresponding Tensor input to the forward loop.
    loop_vars: list of Tensors. The inputs to body_grad_graph.

  Returns:
    The new loop_vars to pass to body_grad_graph.
  """
    init_slices = _create_grad_indexed_slices_init(grad_output_slices, forward_input)
    with body_grad_graph.as_default():
        input_slices = indexed_slices.IndexedSlices(values=body_grad_graph.capture(init_slices.values, allowlisted=True), indices=body_grad_graph.capture(init_slices.indices, allowlisted=True), dense_shape=body_grad_graph.capture(init_slices.dense_shape, allowlisted=True))
        for t in _flatten(init_slices):
            captured_t = body_grad_graph.captures.pop(t)
            body_grad_graph.inputs.remove(captured_t)
        new_output_slices = _rewrite_grad_indexed_slices_output(grad_output_slices, input_slices)
    return _update_indexed_slices_param(body_grad_graph, loop_vars, init_slices, input_slices, new_output_slices, grad_output_slices)