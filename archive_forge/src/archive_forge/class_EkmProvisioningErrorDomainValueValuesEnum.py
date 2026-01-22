from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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