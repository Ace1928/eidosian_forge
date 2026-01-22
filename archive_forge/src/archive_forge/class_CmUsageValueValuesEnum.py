from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CmUsageValueValuesEnum(_messages.Enum):
    """Indicates if and how Container Manager is being used for task
    execution.

    Values:
      CONFIG_NONE: Container Manager is disabled or not running for this
        execution.
      CONFIG_MATCH: Container Manager is enabled and there was a matching
        container available for use during execution.
      CONFIG_MISMATCH: Container Manager is enabled, but there was no matching
        container available for execution.
    """
    CONFIG_NONE = 0
    CONFIG_MATCH = 1
    CONFIG_MISMATCH = 2