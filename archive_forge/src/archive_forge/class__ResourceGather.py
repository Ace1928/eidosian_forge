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
class _ResourceGather(_Node):
    """Specialization of _Node to ResourceGather."""

    def convert_variable_to_constant(self, incoming_edge, tensor_data):
        if self._function is not None:
            return
        if self._node.attr['batch_dims'].i != 0:
            raise ValueError(f"batch_dims must be 0 for freeze_graph, but got node({self._node.name}).attr('batch_dims') = {self._node.attr['batch_dims'].i}.")
        axis_node_name = self._node.name + '/axis'
        axis_dtype = self._node.attr['Tindices']
        axis_data = np.array(self._node.attr['batch_dims'].i)
        converted_graph = self._enclosing_graph.converted_self()
        if axis_node_name not in converted_graph.nodes:
            converted_graph.nodes[axis_node_name] = _Node.new(node=converted_graph.graph_def.node.add(), function=self._function, enclosing_graph=converted_graph)
        output_axis_node = converted_graph.nodes[axis_node_name].node
        output_axis_node.name = axis_node_name
        output_axis_node.op = 'Const'
        output_axis_node.attr['dtype'].CopyFrom(axis_dtype)
        tensor = tensor_util.make_tensor_proto(axis_data, dtype=axis_dtype.type, shape=axis_data.shape)
        output_axis_node.attr['value'].tensor.CopyFrom(tensor)
        output_node = self.converted_self().node
        output_node.Clear()
        output_node.name = self._node.name
        output_node.op = 'GatherV2'
        output_node.input.extend([self._node.input[0], self._node.input[1], axis_node_name])
        output_node.attr['Tparams'].CopyFrom(self._node.attr['dtype'])
        output_node.attr['Tindices'].CopyFrom(self._node.attr['Tindices'])
        output_node.attr['Taxis'].CopyFrom(axis_dtype)
        if '_class' in self._node.attr:
            output_node.attr['_class'].CopyFrom(self._node.attr['_class'])