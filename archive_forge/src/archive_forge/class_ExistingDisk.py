from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExistingDisk(_messages.Message):
    """Configuration for an existing disk to be attached to the VM.

  Fields:
    disk: If `disk` contains slashes, the Cloud Life Sciences API assumes that
      it is a complete URL for the disk. If `disk` does not contain slashes,
      the Cloud Life Sciences API assumes that the disk is a zonal disk and a
      URL will be generated of the form `zones//disks/`, where `` is the zone
      in which the instance is allocated. The disk must be ext4 formatted. If
      all `Mount` references to this disk have the `read_only` flag set to
      true, the disk will be attached in `read-only` mode and can be shared
      with other instances. Otherwise, the disk will be available for writing
      but cannot be shared.
  """
    disk = _messages.StringField(1)