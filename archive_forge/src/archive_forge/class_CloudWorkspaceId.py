from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudWorkspaceId(_messages.Message):
    """A CloudWorkspaceId is a unique identifier for a cloud workspace. A cloud
  workspace is a place associated with a repo where modified files can be
  stored before they are committed.

  Fields:
    name: The unique name of the workspace within the repo. This is the name
      chosen by the client in the Source API's CreateWorkspace method.
    repoId: The ID of the repo containing the workspace.
  """
    name = _messages.StringField(1)
    repoId = _messages.MessageField('RepoId', 2)