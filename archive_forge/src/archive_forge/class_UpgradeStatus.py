from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeStatus(_messages.Message):
    """UpgradeStatus contains information about upgradeAppliance operation.

  Enums:
    StateValueValuesEnum: The state of the upgradeAppliance operation.

  Fields:
    error: Provides details on the state of the upgrade operation in case of
      an error.
    previousVersion: The version from which we upgraded.
    startTime: The time the operation was started.
    state: The state of the upgradeAppliance operation.
    version: The version to upgrade to.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the upgradeAppliance operation.

    Values:
      STATE_UNSPECIFIED: The state was not sampled by the health checks yet.
      RUNNING: The upgrade has started.
      FAILED: The upgrade failed.
      SUCCEEDED: The upgrade finished successfully.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        FAILED = 2
        SUCCEEDED = 3
    error = _messages.MessageField('Status', 1)
    previousVersion = _messages.StringField(2)
    startTime = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    version = _messages.StringField(5)