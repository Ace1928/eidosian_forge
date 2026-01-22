from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InfrastructureTypeValueValuesEnum(_messages.Enum):
    """Optional. The infrastructure type this Membership is running on.

    Values:
      INFRASTRUCTURE_TYPE_UNSPECIFIED: No type was specified. Some Hub
        functionality may require a type be specified, and will not support
        Memberships with this value.
      ON_PREM: Private infrastructure that is owned or operated by customer.
        This includes GKE distributions such as GKE-OnPrem and GKE-
        OnBareMetal.
      MULTI_CLOUD: Public cloud infrastructure.
    """
    INFRASTRUCTURE_TYPE_UNSPECIFIED = 0
    ON_PREM = 1
    MULTI_CLOUD = 2