import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class StringGaugeCell(object):
    """A single string value stored in an `StringGauge`."""
    __slots__ = ['_cell']

    def __init__(self, cell):
        """Creates a new StringGaugeCell.

    Args:
      cell: A c pointer of TFE_MonitoringStringGaugeCell.
    """
        self._cell = cell

    def set(self, value):
        """Atomically set the value.

    Args:
      value: string value.
    """
        pywrap_tfe.TFE_MonitoringStringGaugeCellSet(self._cell, value)

    def value(self):
        """Retrieves the current value."""
        with c_api_util.tf_buffer() as buffer_:
            pywrap_tfe.TFE_MonitoringStringGaugeCellValue(self._cell, buffer_)
            value = pywrap_tf_session.TF_GetBuffer(buffer_).decode('utf-8')
        return value