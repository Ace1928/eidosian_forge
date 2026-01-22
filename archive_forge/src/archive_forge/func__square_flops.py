import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Square', 'flops')
def _square_flops(graph, node):
    """Compute flops for Square operation."""
    return _unary_op_flops(graph, node)