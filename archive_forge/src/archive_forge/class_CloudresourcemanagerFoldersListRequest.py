from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersListRequest(_messages.Message):
    """A CloudresourcemanagerFoldersListRequest object.

  Fields:
    pageSize: The maximum number of Folders to return in the response. This
      field is optional.
    pageToken: A pagination token returned from a previous call to
      `ListFolders` that indicates where this listing should continue from.
      This field is optional.
    parent: The resource name of the Organization or Folder whose Folders are
      being listed. Must be of the form `folders/{folder_id}` or
      `organizations/{org_id}`. Access to this method is controlled by
      checking the `resourcemanager.folders.list` permission on the `parent`.
    showDeleted: Controls whether Folders in the DELETE_REQUESTED state should
      be returned. Defaults to false. This field is optional.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3)
    showDeleted = _messages.BooleanField(4)