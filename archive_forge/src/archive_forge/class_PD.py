from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PD(_messages.Message):
    """Deprecated: please use device_name instead.

  Fields:
    device: PD device name, e.g. persistent-disk-1.
    disk: PD disk name, e.g. pd-1.
    existing: Whether this is an existing PD. Default is false. If false,
      i.e., new PD, we will format it into ext4 and mount to the given path.
      If true, i.e., existing PD, it should be in ext4 format and we will
      mount it to the given path.
  """
    device = _messages.StringField(1)
    disk = _messages.StringField(2)
    existing = _messages.BooleanField(3)