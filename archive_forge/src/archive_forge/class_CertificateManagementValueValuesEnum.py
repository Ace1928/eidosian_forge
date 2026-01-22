from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateManagementValueValuesEnum(_messages.Enum):
    """Specifies workload certificate management.

    Values:
      CERTIFICATE_MANAGEMENT_UNSPECIFIED: Disable workload certificate
        feature.
      DISABLED: Disable workload certificate feature.
      ENABLED: Enable workload certificate feature.
    """
    CERTIFICATE_MANAGEMENT_UNSPECIFIED = 0
    DISABLED = 1
    ENABLED = 2