from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateChangeValueValuesEnum(_messages.Enum):
    """State change of the finding between the points in time.

    Values:
      UNUSED: State change is unused, this is the canonical default for this
        enum.
      CHANGED: The finding has changed state in some way between the points in
        time and existed at both points.
      UNCHANGED: The finding has not changed state between the points in time
        and existed at both points.
      ADDED: The finding was created between the points in time.
      REMOVED: The finding at timestamp does not match the filter specified,
        but it did at timestamp - compare_duration.
    """
    UNUSED = 0
    CHANGED = 1
    UNCHANGED = 2
    ADDED = 3
    REMOVED = 4