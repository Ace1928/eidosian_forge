from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResolvedStateValueValuesEnum(_messages.Enum):
    """The resolved resource's state

    Values:
      RESOLVED_STATE_UNSPECIFIED: Not used
      INFO_DENIED: The caller doesn't have permission to resolve this resource
      COMPLETED: The resource has been fully resolved
      NOT_APPLICABLE: The resource cannot be restricted by service perimeters
      ERROR: The resource cannot be resolved due to an error.
    """
    RESOLVED_STATE_UNSPECIFIED = 0
    INFO_DENIED = 1
    COMPLETED = 2
    NOT_APPLICABLE = 3
    ERROR = 4