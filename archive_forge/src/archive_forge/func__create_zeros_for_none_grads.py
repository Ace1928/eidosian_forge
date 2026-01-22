import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _create_zeros_for_none_grads(forward_graphs, grad_graphs):
    """Creates zeros for None out grads if at least one branch has non-None grad.

  Args:
    forward_graphs: List of forward FuncGraphs.
    grad_graphs: List of grad FuncGraphs.
  """
    assert len(forward_graphs) == len(grad_graphs)
    branch_outputs = [g.structured_outputs for g in grad_graphs]
    num_outputs_per_branch = [len(outs) for outs in branch_outputs]
    assert len(set(num_outputs_per_branch)) == 1, num_outputs_per_branch
    for output_idx, branch_outs in enumerate(zip(*branch_outputs)):
        if any((t is None for t in branch_outs)) and any((t is not None for t in branch_outs)):
            for branch_index, t in enumerate(branch_outs):
                if t is None:
                    with grad_graphs[branch_index].as_default():
                        zeros = default_gradient.zeros_like(forward_graphs[branch_index].inputs[output_idx])
                        grad_graphs[branch_index].structured_outputs[output_idx] = zeros
    for grad_graph in grad_graphs:
        grad_graph.outputs = [t for t in func_graph_module.flatten(grad_graph.structured_outputs) if t is not None]