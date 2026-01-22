import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def _write_report(self, content):
    """Writes the given content to the report."""
    line = '%s %s' % (_TRACER_LOG_PREFIX, content)
    if self._report_file:
        self._report_file.write(line)
    else:
        logging.info(line)