from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsageValueValuesEnum(_messages.Enum):
    """Specifies whether NAT IP is currently serving at least one endpoint or
    not.

    Values:
      IN_USE: <no description>
      UNUSED: <no description>
    """
    IN_USE = 0
    UNUSED = 1