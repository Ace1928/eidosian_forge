from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedGrantTypesValueListEntryValuesEnum(_messages.Enum):
    """AllowedGrantTypesValueListEntryValuesEnum enum type.

    Values:
      GRANT_TYPE_UNSPECIFIED: should not be used
      AUTHORIZATION_CODE_GRANT: authorization code grant
      REFRESH_TOKEN_GRANT: refresh token grant
    """
    GRANT_TYPE_UNSPECIFIED = 0
    AUTHORIZATION_CODE_GRANT = 1
    REFRESH_TOKEN_GRANT = 2