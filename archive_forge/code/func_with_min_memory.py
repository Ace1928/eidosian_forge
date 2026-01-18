import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_min_memory(self, min_bytes=0, min_peak_bytes=0, min_residual_bytes=0, min_output_bytes=0):
    """Only show profiler nodes consuming no less than 'min_bytes'.

    Args:
      min_bytes: Only show profiler nodes requested to allocate no less bytes
          than this.
      min_peak_bytes: Only show profiler nodes using no less than this bytes
          at peak (high watermark). For profiler nodes consist of multiple
          graph nodes, it sums the graph nodes' peak_bytes.
      min_residual_bytes: Only show profiler nodes have no less than
          this bytes not being de-allocated after Compute() ends. For
          profiler nodes consist of multiple graph nodes, it sums the
          graph nodes' residual_bytes.
      min_output_bytes: Only show profiler nodes have no less than this bytes
          output. The output are not necessarily allocated by this profiler
          nodes.
    Returns:
      self
    """
    self._options['min_bytes'] = min_bytes
    self._options['min_peak_bytes'] = min_peak_bytes
    self._options['min_residual_bytes'] = min_residual_bytes
    self._options['min_output_bytes'] = min_output_bytes
    return self