from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InitiatorValueValuesEnum(_messages.Enum):
    """Output only. Whether this ChangeQuorum is a Google or User initiated.

    Values:
      INITIATOR_UNSPECIFIED: Unspecified.
      GOOGLE: ChangeQuorum initiated by Google.
      USER: ChangeQuorum initiated by User.
    """
    INITIATOR_UNSPECIFIED = 0
    GOOGLE = 1
    USER = 2