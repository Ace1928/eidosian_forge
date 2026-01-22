from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NFS(_messages.Message):
    """Represents an NFS volume.

  Fields:
    remotePath: Remote source path exported from the NFS, e.g., "/share".
    server: The IP address of the NFS.
  """
    remotePath = _messages.StringField(1)
    server = _messages.StringField(2)