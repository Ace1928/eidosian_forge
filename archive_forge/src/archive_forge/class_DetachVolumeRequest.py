from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetachVolumeRequest(_messages.Message):
    """Message for detaching Volume from an instance. All Luns of the Volume
  will be detached.

  Fields:
    skipReboot: If true, performs Volume unmapping without instance reboot.
    volume: Name of the Volume to detach.
    volumes: Names of the multiple Volumes to detach. The detaching of volumes
      will have no effect on other existing attached volumes.
  """
    skipReboot = _messages.BooleanField(1)
    volume = _messages.StringField(2)
    volumes = _messages.StringField(3, repeated=True)