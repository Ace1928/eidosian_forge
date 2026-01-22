from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ConfigTypesValueListEntryValuesEnum(_messages.Enum):
    """ConfigTypesValueListEntryValuesEnum enum type.

    Values:
      CONFIG_TYPE_UNSPECIFIED: <no description>
      APT: <no description>
      YUM: <no description>
      GOO: <no description>
      WINDOWS_UPDATE: <no description>
      ZYPPER: <no description>
    """
    CONFIG_TYPE_UNSPECIFIED = 0
    APT = 1
    YUM = 2
    GOO = 3
    WINDOWS_UPDATE = 4
    ZYPPER = 5