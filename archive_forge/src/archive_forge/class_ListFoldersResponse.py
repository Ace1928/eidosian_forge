from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFoldersResponse(_messages.Message):
    """The ListFolders response message.

  Fields:
    folders: A possibly paginated list of Folders that are direct descendants
      of the specified parent resource.
    nextPageToken: A pagination token returned from a previous call to
      `ListFolders` that indicates from where listing should continue. This
      field is optional.
  """
    folders = _messages.MessageField('Folder', 1, repeated=True)
    nextPageToken = _messages.StringField(2)