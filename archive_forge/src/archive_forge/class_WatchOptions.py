import abc
import re
import threading
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import errors
from tensorflow.python.framework import stack
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class WatchOptions:
    """Type for return values of watch_fn."""

    def __init__(self, debug_ops=None, node_name_regex_allowlist=None, op_type_regex_allowlist=None, tensor_dtype_regex_allowlist=None, tolerate_debug_op_creation_failures=False):
        """Constructor of WatchOptions: Debug watch options.

    Used as return values of `watch_fn`s.

    Args:
      debug_ops: (`str` or `list of str`) Debug ops to be used.
      node_name_regex_allowlist: Regular-expression allowlist for node_name,
        e.g., `"(weight_[0-9]+|bias_.*)"`
      op_type_regex_allowlist: Regular-expression allowlist for the op type of
        nodes, e.g., `"(Variable|Add)"`.
        If both `node_name_regex_allowlist` and `op_type_regex_allowlist`
        are set, the two filtering operations will occur in a logical `AND`
        relation. In other words, a node will be included if and only if it
        hits both allowlists.
      tensor_dtype_regex_allowlist: Regular-expression allowlist for Tensor
        data type, e.g., `"^int.*"`.
        This allowlist operates in logical `AND` relations to the two allowlists
        above.
      tolerate_debug_op_creation_failures: (`bool`) whether debug op creation
        failures (e.g., due to dtype incompatibility) are to be tolerated by not
        throwing exceptions.
    """
        if debug_ops:
            self.debug_ops = debug_ops
        else:
            self.debug_ops = ['DebugIdentity']
        self.node_name_regex_allowlist = node_name_regex_allowlist
        self.op_type_regex_allowlist = op_type_regex_allowlist
        self.tensor_dtype_regex_allowlist = tensor_dtype_regex_allowlist
        self.tolerate_debug_op_creation_failures = tolerate_debug_op_creation_failures

    def __repr__(self):
        return 'WatchOptions(debug_ops=%r, node_name_regex_allowlist=%r, op_type_regex_allowlist=%r, tensor_dtype_regex_allowlist=%r, tolerate_debug_op_creation_failures=%r)' % (self.debug_ops, self.node_name_regex_allowlist, self.op_type_regex_allowlist, self.tensor_dtype_regex_allowlist, self.tolerate_debug_op_creation_failures)