from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceActivationValueValuesEnum(_messages.Enum):
    """Consumer Container denotes if the service is active within a project
    or not. This information could be used to clean up resources in case
    service in DISABLED_FULL i.e. Service is inactive > 30 days.

    Values:
      SERVICE_ACTIVATION_STATUS_UNSPECIFIED: Default Unspecified status
      SERVICE_ACTIVATION_ENABLED: Service is active in the project.
      SERVICE_ACTIVATION_DISABLED: Service is disabled in the project recently
        i.e., within last 24 hours.
      SERVICE_ACTIVATION_DISABLED_FULL: Service has been disabled for
        configured grace_period (default 30 days).
      SERVICE_ACTIVATION_UNKNOWN_REASON: Happens when PSM cannot determine the
        status of service in a project Could happen due to variety of reasons
        like PERMISSION_DENIED or Project got deleted etc.
    """
    SERVICE_ACTIVATION_STATUS_UNSPECIFIED = 0
    SERVICE_ACTIVATION_ENABLED = 1
    SERVICE_ACTIVATION_DISABLED = 2
    SERVICE_ACTIVATION_DISABLED_FULL = 3
    SERVICE_ACTIVATION_UNKNOWN_REASON = 4