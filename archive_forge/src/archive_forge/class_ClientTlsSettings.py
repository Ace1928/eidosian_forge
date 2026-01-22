from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientTlsSettings(_messages.Message):
    """[Deprecated] The client side authentication settings for connection
  originating from the backend service. the backend service.

  Enums:
    ModeValueValuesEnum: Indicates whether connections to this port should be
      secured using TLS. The value of this field determines how TLS is
      enforced. This can be set to one of the following values: DISABLE: Do
      not setup a TLS connection to the backends. SIMPLE: Originate a TLS
      connection to the backends. MUTUAL: Secure connections to the backends
      using mutual TLS by presenting client certificates for authentication.

  Fields:
    clientTlsContext: Configures the mechanism to obtain client-side security
      certificates and identity information. This field is only applicable
      when mode is set to MUTUAL.
    mode: Indicates whether connections to this port should be secured using
      TLS. The value of this field determines how TLS is enforced. This can be
      set to one of the following values: DISABLE: Do not setup a TLS
      connection to the backends. SIMPLE: Originate a TLS connection to the
      backends. MUTUAL: Secure connections to the backends using mutual TLS by
      presenting client certificates for authentication.
    sni: SNI string to present to the server during TLS handshake. This field
      is applicable only when mode is SIMPLE or MUTUAL.
    subjectAltNames: A list of alternate names to verify the subject identity
      in the certificate.If specified, the proxy will verify that the server
      certificate's subject alt name matches one of the specified values. This
      field is applicable only when mode is SIMPLE or MUTUAL.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Indicates whether connections to this port should be secured using
    TLS. The value of this field determines how TLS is enforced. This can be
    set to one of the following values: DISABLE: Do not setup a TLS connection
    to the backends. SIMPLE: Originate a TLS connection to the backends.
    MUTUAL: Secure connections to the backends using mutual TLS by presenting
    client certificates for authentication.

    Values:
      DISABLE: Do not setup a TLS connection to the backends.
      INVALID: <no description>
      MUTUAL: Secure connections to the backends using mutual TLS by
        presenting client certificates for authentication.
      SIMPLE: Originate a TLS connection to the backends.
    """
        DISABLE = 0
        INVALID = 1
        MUTUAL = 2
        SIMPLE = 3
    clientTlsContext = _messages.MessageField('TlsContext', 1)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)
    sni = _messages.StringField(3)
    subjectAltNames = _messages.StringField(4, repeated=True)