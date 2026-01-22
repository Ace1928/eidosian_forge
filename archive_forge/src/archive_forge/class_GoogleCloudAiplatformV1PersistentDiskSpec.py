from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PersistentDiskSpec(_messages.Message):
    """Represents the spec of persistent disk options.

  Fields:
    diskSizeGb: Size in GB of the disk (default is 100GB).
    diskType: Type of the disk (default is "pd-standard"). Valid values: "pd-
      ssd" (Persistent Disk Solid State Drive) "pd-standard" (Persistent Disk
      Hard Disk Drive) "pd-balanced" (Balanced Persistent Disk) "pd-extreme"
      (Extreme Persistent Disk)
  """
    diskSizeGb = _messages.IntegerField(1)
    diskType = _messages.StringField(2)