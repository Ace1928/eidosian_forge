import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_min_parameters(self, min_params):
    """Only show profiler nodes holding no less than 'min_params' parameters.

    'Parameters' normally refers the weights of in TensorFlow variables.
    It reflects the 'capacity' of models.

    Args:
      min_params: Only show profiler nodes holding number parameters
          no less than this.
    Returns:
      self
    """
    self._options['min_params'] = min_params
    return self