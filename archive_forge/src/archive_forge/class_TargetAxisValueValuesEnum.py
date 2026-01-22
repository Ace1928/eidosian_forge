from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetAxisValueValuesEnum(_messages.Enum):
    """The target axis to use for plotting the threshold. Target axis is not
    allowed in a Scorecard.

    Values:
      TARGET_AXIS_UNSPECIFIED: The target axis was not specified. Defaults to
        Y1.
      Y1: The y_axis (the right axis of chart).
      Y2: The y2_axis (the left axis of chart).
    """
    TARGET_AXIS_UNSPECIFIED = 0
    Y1 = 1
    Y2 = 2