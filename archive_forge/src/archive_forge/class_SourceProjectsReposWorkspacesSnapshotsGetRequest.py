from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposWorkspacesSnapshotsGetRequest(_messages.Message):
    """A SourceProjectsReposWorkspacesSnapshotsGetRequest object.

  Fields:
    name: The unique name of the workspace within the repo.  This is the name
      chosen by the client in the Source API's CreateWorkspace method.
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
    snapshotId: The ID of the snapshot to get. If empty, the most recent
      snapshot is retrieved.
    workspaceId_repoId_uid: A server-assigned, globally unique identifier.
  """
    name = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    repoName = _messages.StringField(3, required=True)
    snapshotId = _messages.StringField(4, required=True)
    workspaceId_repoId_uid = _messages.StringField(5)