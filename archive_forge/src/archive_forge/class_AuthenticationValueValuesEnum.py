from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthenticationValueValuesEnum(_messages.Enum):
    """AuthenticationValueValuesEnum enum type.

    Values:
      AUTHENTICATION_UNSPECIFIED: <no description>
      AUTHENTICATION_MULTIPLE: <no description>
      AUTHENTICATION_SINGLE: <no description>
      AUTHENTICATION_NONE: <no description>
    """
    AUTHENTICATION_UNSPECIFIED = 0
    AUTHENTICATION_MULTIPLE = 1
    AUTHENTICATION_SINGLE = 2
    AUTHENTICATION_NONE = 3