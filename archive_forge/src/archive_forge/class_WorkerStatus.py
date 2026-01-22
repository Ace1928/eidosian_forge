from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerStatus(_messages.Message):
    """The status of the worker VM.

  Messages:
    AttachedDisksValue: Status of attached disks.

  Fields:
    attachedDisks: Status of attached disks.
    bootDisk: Status of the boot disk.
    freeRamBytes: Free RAM.
    totalRamBytes: Total RAM.
    uptimeSeconds: System uptime.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttachedDisksValue(_messages.Message):
        """Status of attached disks.

    Messages:
      AdditionalProperty: An additional property for a AttachedDisksValue
        object.

    Fields:
      additionalProperties: Additional properties of type AttachedDisksValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttachedDisksValue object.

      Fields:
        key: Name of the additional property.
        value: A DiskStatus attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('DiskStatus', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attachedDisks = _messages.MessageField('AttachedDisksValue', 1)
    bootDisk = _messages.MessageField('DiskStatus', 2)
    freeRamBytes = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    totalRamBytes = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    uptimeSeconds = _messages.IntegerField(5)