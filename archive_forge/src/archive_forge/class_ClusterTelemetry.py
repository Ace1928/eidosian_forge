from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterTelemetry(_messages.Message):
    """Telemetry integration for the cluster.

  Enums:
    TypeValueValuesEnum: Type of the integration.

  Fields:
    type: Type of the integration.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of the integration.

    Values:
      UNSPECIFIED: Not set.
      DISABLED: Monitoring integration is disabled.
      ENABLED: Monitoring integration is enabled.
      SYSTEM_ONLY: Only system components are monitored and logged.
    """
        UNSPECIFIED = 0
        DISABLED = 1
        ENABLED = 2
        SYSTEM_ONLY = 3
    type = _messages.EnumField('TypeValueValuesEnum', 1)