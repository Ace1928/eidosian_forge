import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_min_occurrence(self, min_occurrence):
    """Only show profiler nodes including no less than 'min_occurrence' graph nodes.

    A "node" means a profiler output node, which can be a python line
    (code view), an operation type (op view), or a graph node
    (graph/scope view). A python line includes all graph nodes created by that
    line, while an operation type includes all graph nodes of that type.

    Args:
      min_occurrence: Only show nodes including no less than this.
    Returns:
      self
    """
    self._options['min_occurrence'] = min_occurrence
    return self