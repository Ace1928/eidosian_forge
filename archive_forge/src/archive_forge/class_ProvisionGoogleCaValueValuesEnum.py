from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProvisionGoogleCaValueValuesEnum(_messages.Enum):
    """Immutable. Specifies CA configuration.

    Values:
      GOOGLE_CA_PROVISIONING_UNSPECIFIED: Disable default Google managed CA.
      DISABLED: Disable default Google managed CA.
      ENABLED: Use default Google managed CA.
      ENABLED_WITH_MANAGED_CA: Workload certificate feature is enabled, and
        the entire certificate provisioning process is managed by Google with
        managed CAS which is more secure than the default CA.
      ENABLED_WITH_DEFAULT_CA: Workload certificate feature is enabled, and
        the entire certificate provisioning process is using the default CA
        which is free.
    """
    GOOGLE_CA_PROVISIONING_UNSPECIFIED = 0
    DISABLED = 1
    ENABLED = 2
    ENABLED_WITH_MANAGED_CA = 3
    ENABLED_WITH_DEFAULT_CA = 4