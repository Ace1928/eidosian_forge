from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoveFolderRequest(_messages.Message):
    """The MoveFolder request message.

  Fields:
    destinationParent: The resource name of the Folder or Organization to
      reparent the folder under. Must be of the form `folders/{folder_id}` or
      `organizations/{org_id}`.
  """
    destinationParent = _messages.StringField(1)