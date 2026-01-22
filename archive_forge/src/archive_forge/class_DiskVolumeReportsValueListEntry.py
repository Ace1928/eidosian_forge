from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskVolumeReportsValueListEntry(_messages.Message):
    """A DiskVolumeReportsValueListEntry object.

    Messages:
      VolumeInfoValueListEntry: A VolumeInfoValueListEntry object.

    Fields:
      volumeInfo: Disk volumes
    """

    class VolumeInfoValueListEntry(_messages.Message):
        """A VolumeInfoValueListEntry object.

      Fields:
        storageFree: Free disk space [in bytes]
        storageTotal: Total disk space [in bytes]
        volumeId: Volume id
      """
        storageFree = _messages.IntegerField(1)
        storageTotal = _messages.IntegerField(2)
        volumeId = _messages.StringField(3)
    volumeInfo = _messages.MessageField('VolumeInfoValueListEntry', 1, repeated=True)