import copy
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _get_colocated_node_name(colocated_node_name):
    """Decodes colocated node name and returns it without loc:@ prepended."""
    colocated_node_decoded = colocated_node_name.decode('utf-8')
    if colocated_node_decoded.startswith('loc:@'):
        return colocated_node_decoded[5:]
    return colocated_node_decoded