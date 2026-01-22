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
def _update_indexed_slices_param(graph, loop_vars, init_slices, input_slices, output_slices, old_output_slices):
    """Updates graph with new IndexedSlices input/output.

  Updates graph's metadata to output the gradient computation defined by
  init_slices, input_slices, and output_slices, instead of outputting
  old_output_slices. Also returns a new version of loop_vars with init_slices
  replacing the old input.

  Args:
    graph: _WhileBodyGradFuncGraph.
    loop_vars: the inputs to graph.
    init_slices: the new IndexedSlices to use as input to graph.
    input_slices: the new IndexedSlices in graph that should be fed by
      init_slices.
    output_slices: the new IndexedSlices in graph that should be the
      corresponding output to input_slices.
    old_output_slices: the IndexedSlices in graph that are currently being
      output.

  Returns:
    New loop_vars to pass to graph.
  """
    structured_idx = _get_tensor_index_in_iterable(graph.structured_outputs, old_output_slices)
    flat_idx = _get_tensor_index_in_iterable(graph.outputs, func_graph.flatten(old_output_slices)[0])
    graph.structured_outputs[structured_idx] = output_slices
    graph.outputs = func_graph.flatten(graph.structured_outputs)
    graph.inputs = graph.inputs[:flat_idx] + _flatten(input_slices) + graph.inputs[flat_idx + 1:]
    return loop_vars[:flat_idx] + _flatten(init_slices) + loop_vars[flat_idx + 1:]