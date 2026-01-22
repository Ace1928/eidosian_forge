from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NFSVolumeSource(_messages.Message):
    """Represents a persistent volume that will be mounted using NFS. This
  volume will be shared between all instances of the resource and data will
  not be deleted when the instance is shut down.

  Fields:
    path: Path that is exported by the NFS server.
    readOnly: If true, mount the NFS volume as read only. Defaults to false.
    server: Hostname or IP address of the NFS server.
  """
    path = _messages.StringField(1)
    readOnly = _messages.BooleanField(2)
    server = _messages.StringField(3)