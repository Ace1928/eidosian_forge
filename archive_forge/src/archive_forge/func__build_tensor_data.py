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
def _build_tensor_data(self):
    """Caches the tensor data for all Placeholders in the given function."""
    map_index_to_variable = {}
    for var in self._func.graph.variables:
        for idx, captured_input in enumerate(self._func.captured_inputs):
            if var.handle is captured_input:
                map_index_to_variable[idx] = var
                break
    for idx, (val_tensor, name_tensor) in enumerate(self._func.graph.captures):
        tensor_name = name_tensor.name.split(':')[0]
        if not self._should_convert(tensor_name):
            continue
        if idx in map_index_to_variable:
            data = self._eval(map_index_to_variable[idx])
        else:
            if val_tensor.dtype == dtypes.resource:
                logging.vlog(1, 'Skip converting resource tensor %s' % tensor_name)
                continue
            data = np.array(self._eval(val_tensor))
        self._tensor_data[tensor_name] = _TensorData(numpy=data, dtype=dtypes.as_dtype(data.dtype).as_datatype_enum, index=idx)
    for node in self.node_defs.values():
        if node.op == 'VariableV2':
            if not self._should_convert(node.name):
                continue
            if node.name not in self.tensor_data:
                with self._func.graph.as_default():
                    identity_node = array_ops.identity(self._func.graph.as_graph_element(node.name + ':0'))
                pruned_graph = self._func.prune([], [identity_node.name])()[0]
                self._tensor_data[node.name] = _TensorData(numpy=pruned_graph.numpy(), dtype=node.attr['dtype'].type, index=None)