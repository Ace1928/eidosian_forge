import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('L2Loss', 'flops')
def _l2_loss_flops(graph, node):
    """Compute flops for L2Loss operation."""
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    return ops.OpStats('flops', in_shape.num_elements() * 3 - 1)