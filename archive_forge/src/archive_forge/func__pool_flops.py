import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
def _pool_flops(graph, node):
    """Common code which compute flops for pooling operations."""
    _verify_conv_data_format(node)
    out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    out_shape.assert_is_fully_defined()
    kernel_shape = list(node.attr['ksize'].list.i)
    kernel_area = _list_product(kernel_shape)
    return ops.OpStats('flops', kernel_area * out_shape.num_elements())