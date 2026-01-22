import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('NotEqual', 'flops')
def _not_equal_flops(graph, node):
    """Compute flops for NotEqual operation."""
    return _binary_per_element_op_flops(graph, node)