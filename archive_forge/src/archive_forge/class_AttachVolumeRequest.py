from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttachVolumeRequest(_messages.Message):
    """Message for attaching Volume to an instance. All Luns of the Volume will
  be attached.

  Fields:
    volume: Name of the Volume to attach.
    volumes: Names of the multiple Volumes to attach. The volumes attaching
      will be an additive operation and will have no effect on existing
      attached volumes.
  """
    volume = _messages.StringField(1)
    volumes = _messages.StringField(2, repeated=True)