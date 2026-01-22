from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EphemeralDirectory(_messages.Message):
    """An ephemeral directory which won't persist across workstation sessions.
  It is freshly created on every workstation start operation.

  Fields:
    gcePd: An EphemeralDirectory backed by a Compute Engine persistent disk.
    mountPath: Required. Location of this directory in the running
      workstation.
  """
    gcePd = _messages.MessageField('GcePersistentDisk', 1)
    mountPath = _messages.StringField(2)