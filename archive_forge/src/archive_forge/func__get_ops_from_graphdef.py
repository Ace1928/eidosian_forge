import json
import os
import sys
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_kernel_registry
def _get_ops_from_graphdef(graph_def):
    """Gets the ops and kernels needed from the tensorflow model."""
    ops = set()
    ops.update(_get_ops_from_nodedefs(graph_def.node))
    for function in graph_def.library.function:
        ops.update(_get_ops_from_nodedefs(function.node_def))
    return ops