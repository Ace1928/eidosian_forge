from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateAndAttachVolumeRequest(_messages.Message):
    """Message for creating a volume with immediate Luns allocation and their
  attachment to instances.

  Fields:
    instances: List of instance to attach this volume to. If defined, will
      attach all LUNs of this Volume to specified instances. Makes sense only
      when lun_ranges are defined.
    lunRanges: LUN ranges to be allocated. If defined, will immediately
      allocate LUNs.
    volume: Required. The volume to create.
  """
    instances = _messages.StringField(1, repeated=True)
    lunRanges = _messages.MessageField('VolumeLunRange', 2, repeated=True)
    volume = _messages.MessageField('Volume', 3)