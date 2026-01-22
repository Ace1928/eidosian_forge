from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FastIPMoveValueValuesEnum(_messages.Enum):
    """Enabling fastIPMove is not supported.

    Values:
      DISABLED: <no description>
      GARP_RA: <no description>
    """
    DISABLED = 0
    GARP_RA = 1