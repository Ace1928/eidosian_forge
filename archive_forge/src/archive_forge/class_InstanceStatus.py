from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceStatus(_messages.Message):
    """VM instance status.

  Enums:
    ProvisioningModelValueValuesEnum: The VM instance provisioning model.

  Fields:
    bootDisk: The VM boot disk.
    machineType: The Compute Engine machine type.
    provisioningModel: The VM instance provisioning model.
    taskPack: The max number of tasks can be assigned to this instance type.
  """

    class ProvisioningModelValueValuesEnum(_messages.Enum):
        """The VM instance provisioning model.

    Values:
      PROVISIONING_MODEL_UNSPECIFIED: Unspecified.
      STANDARD: Standard VM.
      SPOT: SPOT VM.
      PREEMPTIBLE: Preemptible VM (PVM). Above SPOT VM is the preferable model
        for preemptible VM instances: the old preemptible VM model (indicated
        by this field) is the older model, and has been migrated to use the
        SPOT model as the underlying technology. This old model will still be
        supported.
    """
        PROVISIONING_MODEL_UNSPECIFIED = 0
        STANDARD = 1
        SPOT = 2
        PREEMPTIBLE = 3
    bootDisk = _messages.MessageField('Disk', 1)
    machineType = _messages.StringField(2)
    provisioningModel = _messages.EnumField('ProvisioningModelValueValuesEnum', 3)
    taskPack = _messages.IntegerField(4)