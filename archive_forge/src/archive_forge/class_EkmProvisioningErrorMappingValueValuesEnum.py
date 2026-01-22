from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EkmProvisioningErrorMappingValueValuesEnum(_messages.Enum):
    """Detailed error message if Ekm provisioning fails

    Values:
      EKM_PROVISIONING_ERROR_MAPPING_UNSPECIFIED: Error is unspecified.
      INVALID_SERVICE_ACCOUNT: Service account is used is invalid.
      MISSING_METRICS_SCOPE_ADMIN_PERMISSION: Iam permission
        monitoring.MetricsScopeAdmin wasn't applied.
      MISSING_EKM_CONNECTION_ADMIN_PERMISSION: Iam permission
        cloudkms.ekmConnectionsAdmin wasn't applied.
    """
    EKM_PROVISIONING_ERROR_MAPPING_UNSPECIFIED = 0
    INVALID_SERVICE_ACCOUNT = 1
    MISSING_METRICS_SCOPE_ADMIN_PERMISSION = 2
    MISSING_EKM_CONNECTION_ADMIN_PERMISSION = 3