from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateModeValueValuesEnum(_messages.Enum):
    """The mode of the certificate.

    Values:
      CERTIFICATE_MODE_UNSPECIFIED: <no description>
      NONE: Do not provision an HTTPS certificate.
      AUTOMATIC: Automatically provisions an HTTPS certificate via GoogleCA.
    """
    CERTIFICATE_MODE_UNSPECIFIED = 0
    NONE = 1
    AUTOMATIC = 2