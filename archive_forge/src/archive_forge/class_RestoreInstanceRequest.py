from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreInstanceRequest(_messages.Message):
    """Request options for restoring an instance

  Fields:
    backup: Required. Backup being used to restore the instance Format: projec
      ts/{project}/locations/{location}/instances/{instance}/backups/{backup}
  """
    backup = _messages.StringField(1)