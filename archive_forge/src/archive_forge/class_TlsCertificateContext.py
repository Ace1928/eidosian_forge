from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsCertificateContext(_messages.Message):
    """[Deprecated] Defines the mechanism to obtain the client or server
  certificate. Defines the mechanism to obtain the client or server
  certificate.

  Enums:
    CertificateSourceValueValuesEnum: Defines how TLS certificates are
      obtained.

  Fields:
    certificatePaths: Specifies the certificate and private key paths. This
      field is applicable only if tlsCertificateSource is set to USE_PATH.
    certificateSource: Defines how TLS certificates are obtained.
    sdsConfig: Specifies the config to retrieve certificates through SDS. This
      field is applicable only if tlsCertificateSource is set to USE_SDS.
  """

    class CertificateSourceValueValuesEnum(_messages.Enum):
        """Defines how TLS certificates are obtained.

    Values:
      INVALID: <no description>
      USE_PATH: USE_PATH specifies that the certificates and private key are
        obtained from a locally mounted filesystem path.
      USE_SDS: USE_SDS specifies that the certificates and private key are
        obtained from a SDS server.
    """
        INVALID = 0
        USE_PATH = 1
        USE_SDS = 2
    certificatePaths = _messages.MessageField('TlsCertificatePaths', 1)
    certificateSource = _messages.EnumField('CertificateSourceValueValuesEnum', 2)
    sdsConfig = _messages.MessageField('SdsConfig', 3)