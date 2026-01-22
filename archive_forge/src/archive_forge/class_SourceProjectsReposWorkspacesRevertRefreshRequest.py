from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposWorkspacesRevertRefreshRequest(_messages.Message):
    """A SourceProjectsReposWorkspacesRevertRefreshRequest object.

  Fields:
    name: The unique name of the workspace within the repo.  This is the name
      chosen by the client in the Source API's CreateWorkspace method.
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
    revertRefreshRequest: A RevertRefreshRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    repoName = _messages.StringField(3, required=True)
    revertRefreshRequest = _messages.MessageField('RevertRefreshRequest', 4)