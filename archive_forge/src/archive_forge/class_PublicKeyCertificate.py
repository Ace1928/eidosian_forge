from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublicKeyCertificate(_messages.Message):
    """A public key certificate format and data.

  Enums:
    FormatValueValuesEnum: The certificate format.

  Fields:
    certificate: The certificate data.
    format: The certificate format.
    x509Details: [Output only] The certificate details. Used only for X.509
      certificates.
  """

    class FormatValueValuesEnum(_messages.Enum):
        """The certificate format.

    Values:
      UNSPECIFIED_PUBLIC_KEY_CERTIFICATE_FORMAT: The format has not been
        specified. This is an invalid default value and must not be used.
      X509_CERTIFICATE_PEM: An X.509v3 certificate
        ([RFC5280](https://www.ietf.org/rfc/rfc5280.txt)), encoded in base64,
        and wrapped by `-----BEGIN CERTIFICATE-----` and `-----END
        CERTIFICATE-----`.
    """
        UNSPECIFIED_PUBLIC_KEY_CERTIFICATE_FORMAT = 0
        X509_CERTIFICATE_PEM = 1
    certificate = _messages.StringField(1)
    format = _messages.EnumField('FormatValueValuesEnum', 2)
    x509Details = _messages.MessageField('X509CertificateDetails', 3)