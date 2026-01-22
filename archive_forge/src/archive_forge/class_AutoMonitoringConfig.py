from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoMonitoringConfig(_messages.Message):
    """AutoMonitoringConfig defines the configuration for GKE Workload Auto-
  Monitoring.

  Enums:
    ScopeValueValuesEnum: Scope for GKE Workload Auto-Monitoring.

  Fields:
    scope: Scope for GKE Workload Auto-Monitoring.
  """

    class ScopeValueValuesEnum(_messages.Enum):
        """Scope for GKE Workload Auto-Monitoring.

    Values:
      SCOPE_UNSPECIFIED: Not set.
      ALL: Auto-Monitoring is enabled for all supported applications.
      NONE: Disable Auto-Monitoring.
    """
        SCOPE_UNSPECIFIED = 0
        ALL = 1
        NONE = 2
    scope = _messages.EnumField('ScopeValueValuesEnum', 1)