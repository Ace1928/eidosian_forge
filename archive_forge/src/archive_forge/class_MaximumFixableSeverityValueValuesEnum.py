from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaximumFixableSeverityValueValuesEnum(_messages.Enum):
    """Required. The threshold for severity for which a fix is currently
    available. This field is required and must be set.

    Values:
      MAXIMUM_ALLOWED_SEVERITY_UNSPECIFIED: Not specified.
      BLOCK_ALL: Block any vulnerability.
      MINIMAL: Allow only minimal severity.
      LOW: Allow only low severity and lower.
      MEDIUM: Allow medium severity and lower.
      HIGH: Allow high severity and lower.
      CRITICAL: Allow critical severity and lower.
      ALLOW_ALL: Allow all severity, even vulnerability with unspecified
        severity.
    """
    MAXIMUM_ALLOWED_SEVERITY_UNSPECIFIED = 0
    BLOCK_ALL = 1
    MINIMAL = 2
    LOW = 3
    MEDIUM = 4
    HIGH = 5
    CRITICAL = 6
    ALLOW_ALL = 7