import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class StringGauge(Metric):
    """A stateful class for updating a gauge-like string metric.

  This class encapsulates a set of string values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """
    __slots__ = []

    def __init__(self, name, description, *labels):
        """Creates a new StringGauge.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
        super(StringGauge, self).__init__('StringGauge', _string_gauge_methods, len(labels), name, description, *labels)

    def get_cell(self, *labels):
        """Retrieves the cell."""
        return StringGaugeCell(super(StringGauge, self).get_cell(*labels))