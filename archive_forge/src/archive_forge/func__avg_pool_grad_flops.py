import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('AvgPoolGrad', 'flops')
def _avg_pool_grad_flops(graph, node):
    """Compute flops for AvgPoolGrad operation."""
    _verify_conv_data_format(node)
    out_backprop_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[1])
    out_backprop_shape.assert_is_fully_defined()
    kernel_shape = list(node.attr['ksize'].list.i)
    kernel_area = _list_product(kernel_shape)
    return ops.OpStats('flops', kernel_area * out_backprop_shape.num_elements() * 2)