from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersMoveRequest(_messages.Message):
    """A CloudresourcemanagerFoldersMoveRequest object.

  Fields:
    foldersId: Part of `name`. The resource name of the Folder to move. Must
      be of the form folders/{folder_id}
    moveFolderRequest: A MoveFolderRequest resource to be passed as the
      request body.
  """
    foldersId = _messages.StringField(1, required=True)
    moveFolderRequest = _messages.MessageField('MoveFolderRequest', 2)