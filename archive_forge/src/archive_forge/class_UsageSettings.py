from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsageSettings(_messages.Message):
    """Usage settings for a consumer of a service.

  Enums:
    ConsumerEnableStatusValueValuesEnum: Consumer controlled setting to
      enable/disable use of this service by the consumer project. The default
      value of this is controlled by the service configuration.

  Fields:
    consumerEnableStatus: Consumer controlled setting to enable/disable use of
      this service by the consumer project. The default value of this is
      controlled by the service configuration.
  """

    class ConsumerEnableStatusValueValuesEnum(_messages.Enum):
        """Consumer controlled setting to enable/disable use of this service by
    the consumer project. The default value of this is controlled by the
    service configuration.

    Values:
      DISABLED: The service is disabled.
      ENABLED: The service is enabled.
    """
        DISABLED = 0
        ENABLED = 1
    consumerEnableStatus = _messages.EnumField('ConsumerEnableStatusValueValuesEnum', 1)