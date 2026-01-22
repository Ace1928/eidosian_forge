from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2DiskPath(_messages.Message):
    """Path of the file in terms of underlying disk/partition identifiers.

  Fields:
    partitionUuid: UUID of the partition (format
      https://wiki.archlinux.org/title/persistent_block_device_naming#by-uuid)
    relativePath: Relative path of the file in the partition as a JSON encoded
      string. Example: /home/user1/executable_file.sh
  """
    partitionUuid = _messages.StringField(1)
    relativePath = _messages.StringField(2)