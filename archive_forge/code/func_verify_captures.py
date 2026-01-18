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
def verify_captures(op_type, branch_graphs):
    """Verify that a branch's tensor is not accessed in another branch fn."""
    other_branch_graphs = {g: i for i, g in enumerate(branch_graphs)}
    for i, branch_graph in enumerate(branch_graphs):
        for t in branch_graph.external_captures:
            if not isinstance(t, ops.EagerTensor) and t.graph in other_branch_graphs:
                branch_names = ['true_fn', 'false_fn'] if op_type == _COND else ['branch {}'.format(bi) for bi in range(len(branch_graphs))]
                raise ValueError('Tensor {tname} in {b0name} is accessed from {b1name}.'.format(tname=t.name, b0name=branch_names[other_branch_graphs[t.graph]], b1name=branch_names[i]))