from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InitializeParams(_messages.Message):
    """Specifies the parameters to initialize this disk.

  Fields:
    diskName: Optional. Specifies the disk name. If not specified, the default
      is to use the name of the instance.
    replicaZones: Optional. Required for each regional disk associated with
      the instance.
  """
    diskName = _messages.StringField(1)
    replicaZones = _messages.StringField(2, repeated=True)