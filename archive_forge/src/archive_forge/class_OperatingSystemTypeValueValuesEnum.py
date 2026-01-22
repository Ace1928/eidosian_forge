from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OperatingSystemTypeValueValuesEnum(_messages.Enum):
    """The operating system type where this account applies.

    Values:
      OPERATING_SYSTEM_TYPE_UNSPECIFIED: The operating system type associated
        with the user account information is unspecified.
      LINUX: Linux user account information.
      WINDOWS: Windows user account information.
    """
    OPERATING_SYSTEM_TYPE_UNSPECIFIED = 0
    LINUX = 1
    WINDOWS = 2