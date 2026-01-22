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
def _decorate_run_options_for_debug(self, run_options, debug_urls, debug_ops='DebugIdentity', node_name_regex_allowlist=None, op_type_regex_allowlist=None, tensor_dtype_regex_allowlist=None, tolerate_debug_op_creation_failures=False):
    """Modify a RunOptions object for debug tensor watching.

    Specifies request for outputting partition graphs. Adds
    debug_tensor_watch_opts with proper debug URLs.

    Args:
      run_options: (RunOptions) the modified RunOptions object.
      debug_urls: (list of str) debug URLs to be entered in run_options.
        debug_tensor_watch_opts.
      debug_ops: (str or list of str) debug op(s) to be used by the debugger.
      node_name_regex_allowlist: Regular-expression allowlist for node
        name.
      op_type_regex_allowlist: Regular-expression allowlist for op type.
      tensor_dtype_regex_allowlist: Regular-expression allowlist for tensor
        dtype.
      tolerate_debug_op_creation_failures: Whether debug op creation failures
        are to be tolerated.
    """
    run_options.output_partition_graphs = True
    debug_utils.watch_graph(run_options, self._sess.graph, debug_urls=debug_urls, debug_ops=debug_ops, node_name_regex_allowlist=node_name_regex_allowlist, op_type_regex_allowlist=op_type_regex_allowlist, tensor_dtype_regex_allowlist=tensor_dtype_regex_allowlist, tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures, reset_disk_byte_usage=self._run_call_count == 1 or self._is_disk_usage_reset_each_run())