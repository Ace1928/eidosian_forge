import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
class OpenReportFile(object):
    """Context manager for writing report file."""

    def __init__(self, tt_parameters):
        if not tt_parameters.report_file_path:
            self._report_file = None
            return
        try:
            self._report_file = gfile.Open(tt_parameters.report_file_path, 'w')
        except IOError as e:
            raise e

    def __enter__(self):
        return self._report_file

    def __exit__(self, unused_type, unused_value, unused_traceback):
        if self._report_file:
            self._report_file.close()