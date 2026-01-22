import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('ArgMax', 'flops')
def _arg_max_flops(graph, node):
    """Compute flops for ArgMax operation."""
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)