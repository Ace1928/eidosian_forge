import collections
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
class _VarHandle(_Node):
    """Specialization of _Node to VarHandleOp."""

    def convert_variable_to_constant(self, incoming_edge, tensor_data):
        tensor_proto = tensor_util.make_tensor_proto(tensor_data.numpy, tensor_data.dtype, tensor_data.numpy.shape)
        node = self.converted_self().node
        node.Clear()
        node.name = self._node.name
        node.op = 'Const'
        node.attr['dtype'].CopyFrom(tensor_data.dtype_attr)
        node.attr['value'].tensor.CopyFrom(tensor_proto)
        for edge in self.outgoing_edges:
            edge.destination.convertible.convert_variable_to_constant(edge, tensor_data)