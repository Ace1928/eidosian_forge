import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
@contextlib.contextmanager
def _disable_summary_context():
    """Enters a context where all summary ops are skipped.

  Summaries are not yet supported in xla.compile(). So we provide this context
  manager that can skip creating summary ops. This is a temporary workaround due
  to XLA not supporting summary ops.

  Yields:
    None.
  """
    original_skip_summary_func = summary_op_util.skip_summary
    summary_op_util.skip_summary = lambda: True
    try:
        yield
    finally:
        summary_op_util.skip_summary = original_skip_summary_func