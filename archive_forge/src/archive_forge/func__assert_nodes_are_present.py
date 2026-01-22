import copy
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _assert_nodes_are_present(name_to_node, nodes):
    """Assert that nodes are present in the graph."""
    for d in nodes:
        assert d in name_to_node, '%s is not in graph' % d