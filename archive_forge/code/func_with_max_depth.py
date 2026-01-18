import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_max_depth(self, max_depth):
    """Set the maximum depth of display.

    The depth depends on profiling view. For 'scope' view, it's the
    depth of name scope hierarchy (tree), for 'op' view, it's the number
    of operation types (list), etc.

    Args:
      max_depth: Maximum depth of the data structure to display.
    Returns:
      self
    """
    self._options['max_depth'] = max_depth
    return self