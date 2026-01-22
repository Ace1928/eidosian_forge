import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('AvgPool', 'flops')
def _avg_pool_flops(graph, node):
    """Compute flops for AvgPool operation."""
    return _pool_flops(graph, node)