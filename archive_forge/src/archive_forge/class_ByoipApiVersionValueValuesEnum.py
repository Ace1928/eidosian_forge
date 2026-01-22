from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ByoipApiVersionValueValuesEnum(_messages.Enum):
    """[Output Only] The version of BYOIP API.

    Values:
      V1: This public delegated prefix usually takes 4 weeks to delete, and
        the BGP status cannot be changed. Announce and Withdraw APIs can not
        be used on this prefix.
      V2: This public delegated prefix takes minutes to delete. Announce and
        Withdraw APIs can be used on this prefix to change the BGP status.
    """
    V1 = 0
    V2 = 1