from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MeasurementSelectionTypeValueValuesEnum(_messages.Enum):
    """Describe which measurement selection type will be used

    Values:
      MEASUREMENT_SELECTION_TYPE_UNSPECIFIED: Will be treated as
        LAST_MEASUREMENT.
      LAST_MEASUREMENT: Use the last measurement reported.
      BEST_MEASUREMENT: Use the best measurement reported.
    """
    MEASUREMENT_SELECTION_TYPE_UNSPECIFIED = 0
    LAST_MEASUREMENT = 1
    BEST_MEASUREMENT = 2