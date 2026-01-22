import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class BoolGaugeCell(object):
    """A single boolean value stored in an `BoolGauge`."""
    __slots__ = ['_cell']

    def __init__(self, cell):
        """Creates a new BoolGaugeCell.

    Args:
      cell: A c pointer of TFE_MonitoringBoolGaugeCell.
    """
        self._cell = cell

    def set(self, value):
        """Atomically set the value.

    Args:
      value: bool value.
    """
        pywrap_tfe.TFE_MonitoringBoolGaugeCellSet(self._cell, value)

    def value(self):
        """Retrieves the current value."""
        return pywrap_tfe.TFE_MonitoringBoolGaugeCellValue(self._cell)