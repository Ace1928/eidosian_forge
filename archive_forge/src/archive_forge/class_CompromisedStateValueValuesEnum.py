from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompromisedStateValueValuesEnum(_messages.Enum):
    """Compromised State of the DeviceUser object

    Values:
      COMPROMISED_STATE_UNSPECIFIED: Compromised state of Device User account
        is unknown or unspecified.
      COMPROMISED: Device User Account is compromised.
      NOT_COMPROMISED: Device User Account is not compromised.
    """
    COMPROMISED_STATE_UNSPECIFIED = 0
    COMPROMISED = 1
    NOT_COMPROMISED = 2