import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_empty_output(self):
    """Do not generate side-effect outputs."""
    self._options['output'] = 'none'
    return self