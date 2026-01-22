from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1WorkloadEkmProvisioningResponse(_messages.Message):
    """External key management systems(EKM) Provisioning response

  Enums:
    EkmProvisioningErrorDomainValueValuesEnum: Indicates Ekm provisioning
      error if any.
    EkmProvisioningErrorMappingValueValuesEnum: Detailed error message if Ekm
      provisioning fails
    EkmProvisioningStateValueValuesEnum: Indicates Ekm enrollment Provisioning
      of a given workload.

  Fields:
    ekmProvisioningErrorDomain: Indicates Ekm provisioning error if any.
    ekmProvisioningErrorMapping: Detailed error message if Ekm provisioning
      fails
    ekmProvisioningState: Indicates Ekm enrollment Provisioning of a given
      workload.
  """

    class EkmProvisioningErrorDomainValueValuesEnum(_messages.Enum):
        """Indicates Ekm provisioning error if any.

    Values:
      EKM_PROVISIONING_ERROR_DOMAIN_UNSPECIFIED: No error domain
      UNSPECIFIED_ERROR: Error but domain is unspecified.
      GOOGLE_SERVER_ERROR: Internal logic breaks within provisioning code.
      EXTERNAL_USER_ERROR: Error occurred with the customer not granting
        permission/creating resource.
      EXTERNAL_PARTNER_ERROR: Error occurred within the partner's provisioning
        cluster.
      TIMEOUT_ERROR: Resource wasn't provisioned in the required 7 day time
        period
    """
        EKM_PROVISIONING_ERROR_DOMAIN_UNSPECIFIED = 0
        UNSPECIFIED_ERROR = 1
        GOOGLE_SERVER_ERROR = 2
        EXTERNAL_USER_ERROR = 3
        EXTERNAL_PARTNER_ERROR = 4
        TIMEOUT_ERROR = 5

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

    class EkmProvisioningStateValueValuesEnum(_messages.Enum):
        """Indicates Ekm enrollment Provisioning of a given workload.

    Values:
      EKM_PROVISIONING_STATE_UNSPECIFIED: Default State for Ekm Provisioning
      EKM_PROVISIONING_STATE_PENDING: Pending State for Ekm Provisioning
      EKM_PROVISIONING_STATE_FAILED: Failed State for Ekm Provisioning
      EKM_PROVISIONING_STATE_COMPLETED: Completed State for Ekm Provisioning
    """
        EKM_PROVISIONING_STATE_UNSPECIFIED = 0
        EKM_PROVISIONING_STATE_PENDING = 1
        EKM_PROVISIONING_STATE_FAILED = 2
        EKM_PROVISIONING_STATE_COMPLETED = 3
    ekmProvisioningErrorDomain = _messages.EnumField('EkmProvisioningErrorDomainValueValuesEnum', 1)
    ekmProvisioningErrorMapping = _messages.EnumField('EkmProvisioningErrorMappingValueValuesEnum', 2)
    ekmProvisioningState = _messages.EnumField('EkmProvisioningStateValueValuesEnum', 3)