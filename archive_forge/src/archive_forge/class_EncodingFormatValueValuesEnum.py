from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncodingFormatValueValuesEnum(_messages.Enum):
    """Optional. Specifies the encoding format of each CertificateAuthority
    resource's CA certificate and CRLs. If this is omitted, CA certificates
    and CRLs will be published in PEM.

    Values:
      ENCODING_FORMAT_UNSPECIFIED: Not specified. By default, PEM format will
        be used.
      PEM: The CertificateAuthority's CA certificate and CRLs will be
        published in PEM format.
      DER: The CertificateAuthority's CA certificate and CRLs will be
        published in DER format.
    """
    ENCODING_FORMAT_UNSPECIFIED = 0
    PEM = 1
    DER = 2