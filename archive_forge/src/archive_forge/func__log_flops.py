import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Log', 'flops')
def _log_flops(graph, node):
    """Compute flops for Log operation."""
    return _unary_op_flops(graph, node)