from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthMethodValueValuesEnum(_messages.Enum):
    """Specifies the authentication and authorization method used by the
    storage service. When not specified, Transfer Service will attempt to
    determine right auth method to use.

    Values:
      AUTH_METHOD_UNSPECIFIED: AuthMethod is not specified.
      AUTH_METHOD_AWS_SIGNATURE_V4: Auth requests with AWS SigV4.
      AUTH_METHOD_AWS_SIGNATURE_V2: Auth requests with AWS SigV2.
    """
    AUTH_METHOD_UNSPECIFIED = 0
    AUTH_METHOD_AWS_SIGNATURE_V4 = 1
    AUTH_METHOD_AWS_SIGNATURE_V2 = 2