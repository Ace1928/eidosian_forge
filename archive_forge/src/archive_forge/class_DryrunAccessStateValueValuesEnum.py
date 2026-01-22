from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DryrunAccessStateValueValuesEnum(_messages.Enum):
    """The access state of the dry run service perimeters

    Values:
      ACCESS_STATE_UNSPECIFIED: Not used
      NOT_APPLICABLE: The request is not restricted by any service perimeters
      GRANTED: The request is granted by service perimeters
      DENIED: The request is denied by service perimeters
    """
    ACCESS_STATE_UNSPECIFIED = 0
    NOT_APPLICABLE = 1
    GRANTED = 2
    DENIED = 3