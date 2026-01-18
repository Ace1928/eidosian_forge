import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_file_output(self, outfile):
    """Print the result to a file."""
    self._options['output'] = 'file:outfile=%s' % outfile
    return self