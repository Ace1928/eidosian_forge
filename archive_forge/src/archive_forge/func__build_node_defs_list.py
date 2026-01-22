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
def _build_node_defs_list(self):
    """Builds the list of NodeDefs in the GraphDef.

    This list consists of all NodeDefs in the main graph as well as all control
    flow NodeDefs in the functions.

    The remaining NodeDefs in the functions are not included because the op
    names
    are not unique and the variables are handled differently than the main
    graph.
    The control flow ops need to be extracted because they are need their
    attributes to be updated similar to the control flow ops in the main graph.
    """
    self._node_defs = {node.name: node for node in self._graph_def.node}
    if self._graph_def.library:
        for func in self._graph_def.library.function:
            self._node_defs.update({node.name: node for node in func.node_def if node.op in _CONTROL_FLOW_OPS})