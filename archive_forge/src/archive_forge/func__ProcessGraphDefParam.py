import contextlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _ProcessGraphDefParam(graph_def):
    """Type-checks and possibly canonicalizes `graph_def`."""
    if not isinstance(graph_def, graph_pb2.GraphDef):
        try:
            old_graph_def = graph_def
            graph_def = graph_pb2.GraphDef()
            graph_def.MergeFrom(old_graph_def)
        except TypeError:
            raise TypeError('Argument `graph_def` must be a GraphDef proto.')
    else:
        for node in graph_def.node:
            op_def = op_def_registry.get(node.op)
            if op_def is None:
                continue
            _SetDefaultAttrValues(node, op_def)
    return graph_def