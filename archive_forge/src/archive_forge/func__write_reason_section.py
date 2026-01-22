import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def _write_reason_section(self):
    """Writes the reason section of the report."""
    self._write_report('%s %s\n' % (_MARKER_SECTION_BEGIN, _SECTION_NAME_REASON))
    for key in sorted(self.instrument_records):
        self._write_report('"%s" %s\n' % (key, self.instrument_records[key]))
    self._write_report('%s %s\n' % (_MARKER_SECTION_END, _SECTION_NAME_REASON))