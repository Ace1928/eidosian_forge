import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Conv2DBackpropFilter', 'flops')
def _conv_2d_backprop_filter_flops(graph, node):
    """Compute flops for Conv2DBackpropFilter operation."""
    _verify_conv_data_format(node)
    image_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    image_shape.assert_is_fully_defined()
    kernel_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    kernel_shape.assert_is_fully_defined()
    strides_shape = list(node.attr['strides'].list.i)
    strides_product = strides_shape[1] * strides_shape[2]
    return ops.OpStats('flops', 2 * image_shape.num_elements() * kernel_shape.num_elements() / (image_shape.dims[-1].value * strides_product))