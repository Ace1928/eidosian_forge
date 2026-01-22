from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MutualTls(_messages.Message):
    """[Deprecated] Configuration for the mutual Tls mode for peer
  authentication. Configuration for the mutual Tls mode for peer
  authentication.

  Enums:
    ModeValueValuesEnum: Specifies if the server TLS is configured to be
      strict or permissive. This field can be set to one of the following:
      STRICT: Client certificate must be presented, connection is in TLS.
      PERMISSIVE: Client certificate can be omitted, connection can be either
      plaintext or TLS.

  Fields:
    mode: Specifies if the server TLS is configured to be strict or
      permissive. This field can be set to one of the following: STRICT:
      Client certificate must be presented, connection is in TLS. PERMISSIVE:
      Client certificate can be omitted, connection can be either plaintext or
      TLS.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Specifies if the server TLS is configured to be strict or permissive.
    This field can be set to one of the following: STRICT: Client certificate
    must be presented, connection is in TLS. PERMISSIVE: Client certificate
    can be omitted, connection can be either plaintext or TLS.

    Values:
      INVALID: <no description>
      PERMISSIVE: Client certificate can be omitted, connection can be either
        plaintext or TLS.
      STRICT: Client certificate must be presented, connection is in TLS.
    """
        INVALID = 0
        PERMISSIVE = 1
        STRICT = 2
    mode = _messages.EnumField('ModeValueValuesEnum', 1)