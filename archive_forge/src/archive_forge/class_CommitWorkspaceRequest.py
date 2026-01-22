from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CommitWorkspaceRequest(_messages.Message):
    """Request for CommitWorkspace.

  Fields:
    author: Author of the commit in the format: "Author Name
      <author@example.com>" required
    currentSnapshotId: If non-empty, current_snapshot_id must refer to the
      most recent update to the workspace, or ABORTED is returned.
    message: The commit message. required
    paths: The subset of modified paths to commit. If empty, then commit all
      modified paths.
    workspaceId: The ID of the workspace.
  """
    author = _messages.StringField(1)
    currentSnapshotId = _messages.StringField(2)
    message = _messages.StringField(3)
    paths = _messages.StringField(4, repeated=True)
    workspaceId = _messages.MessageField('CloudWorkspaceId', 5)