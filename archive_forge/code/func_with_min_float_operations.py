import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_min_float_operations(self, min_float_ops):
    """Only show profiler nodes consuming no less than 'min_float_ops'.

    Please see https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md
    on the caveats of calculating float operations.

    Args:
      min_float_ops: Only show profiler nodes with float operations
          no less than this.
    Returns:
      self
    """
    self._options['min_float_ops'] = min_float_ops
    return self