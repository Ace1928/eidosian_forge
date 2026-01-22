from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeEphemeralStorageConfig(_messages.Message):
    """GkeEphemeralStorageConfig contains configuration for the ephemeral
  storage filesystem.

  Fields:
    localSsdCount: Number of local SSDs to use to back ephemeral storage. Uses
      NVMe interfaces. Each local SSD is 375 GB in size. If zero, it means to
      disable using local SSDs as ephemeral storage.
  """
    localSsdCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)