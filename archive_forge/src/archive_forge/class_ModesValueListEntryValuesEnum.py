from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModesValueListEntryValuesEnum(_messages.Enum):
    """ModesValueListEntryValuesEnum enum type.

    Values:
      ADDRESS_MODE_UNSPECIFIED: Internet protocol not set.
      MODE_IPV4: Use the IPv4 internet protocol.
    """
    ADDRESS_MODE_UNSPECIFIED = 0
    MODE_IPV4 = 1