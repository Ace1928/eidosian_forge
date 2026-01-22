from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceWithNamedPorts(_messages.Message):
    """A InstanceWithNamedPorts object.

  Enums:
    StatusValueValuesEnum: [Output Only] The status of the instance.

  Fields:
    instance: [Output Only] The URL of the instance.
    namedPorts: [Output Only] The named ports that belong to this instance
      group.
    status: [Output Only] The status of the instance.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of the instance.

    Values:
      DEPROVISIONING: The instance is halted and we are performing tear down
        tasks like network deprogramming, releasing quota, IP, tearing down
        disks etc.
      PROVISIONING: Resources are being allocated for the instance.
      REPAIRING: The instance is in repair.
      RUNNING: The instance is running.
      STAGING: All required resources have been allocated and the instance is
        being started.
      STOPPED: The instance has stopped successfully.
      STOPPING: The instance is currently stopping (either being deleted or
        killed).
      SUSPENDED: The instance has suspended.
      SUSPENDING: The instance is suspending.
      TERMINATED: The instance has stopped (either by explicit action or
        underlying failure).
    """
        DEPROVISIONING = 0
        PROVISIONING = 1
        REPAIRING = 2
        RUNNING = 3
        STAGING = 4
        STOPPED = 5
        STOPPING = 6
        SUSPENDED = 7
        SUSPENDING = 8
        TERMINATED = 9
    instance = _messages.StringField(1)
    namedPorts = _messages.MessageField('NamedPort', 2, repeated=True)
    status = _messages.EnumField('StatusValueValuesEnum', 3)