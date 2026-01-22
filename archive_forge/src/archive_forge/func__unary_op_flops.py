import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
def _unary_op_flops(graph, node, ops_per_element=1):
    """Common code which compute flops for unary operations."""
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    return ops.OpStats('flops', in_shape.num_elements() * ops_per_element)