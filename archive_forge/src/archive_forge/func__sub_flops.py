import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Sub', 'flops')
def _sub_flops(graph, node):
    """Compute flops for Sub operation."""
    return _binary_per_element_op_flops(graph, node)