from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityTypeValueValuesEnum(_messages.Enum):
    """Specifies the type of identities that are allowed access from outside
    the perimeter. If left unspecified, then members of `identities` field
    will be allowed access.

    Values:
      IDENTITY_TYPE_UNSPECIFIED: No blanket identity group specified.
      ANY_IDENTITY: Authorize access from all identities outside the
        perimeter.
      ANY_USER_ACCOUNT: Authorize access from all human users outside the
        perimeter.
      ANY_SERVICE_ACCOUNT: Authorize access from all service accounts outside
        the perimeter.
    """
    IDENTITY_TYPE_UNSPECIFIED = 0
    ANY_IDENTITY = 1
    ANY_USER_ACCOUNT = 2
    ANY_SERVICE_ACCOUNT = 3