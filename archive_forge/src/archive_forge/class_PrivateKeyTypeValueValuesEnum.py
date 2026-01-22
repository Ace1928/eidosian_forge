from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PrivateKeyTypeValueValuesEnum(_messages.Enum):
    """The output format for the private key. Only provided in
    `CreateServiceAccountKey` responses, not in `GetServiceAccountKey` or
    `ListServiceAccountKey` responses.  Google never exposes system-managed
    private keys, and never retains user-managed private keys.

    Values:
      TYPE_UNSPECIFIED: Unspecified. Equivalent to
        `TYPE_GOOGLE_CREDENTIALS_FILE`.
      TYPE_PKCS12_FILE: PKCS12 format. The password for the PKCS12 file is
        `notasecret`. For more information, see
        https://tools.ietf.org/html/rfc7292.
      TYPE_GOOGLE_CREDENTIALS_FILE: Google Credentials File format.
    """
    TYPE_UNSPECIFIED = 0
    TYPE_PKCS12_FILE = 1
    TYPE_GOOGLE_CREDENTIALS_FILE = 2