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
Updates graph with new IndexedSlices input/output.

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
  