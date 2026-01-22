from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitLabConnectedRepository(_messages.Message):
    """GitLabConnectedRepository represents a GitLab connected repository
  request response.

  Fields:
    parent: The name of the `GitLabConfig` that added connected repository.
      Format: `projects/{project}/locations/{location}/gitLabConfigs/{config}`
    repo: The GitLab repositories to connect.
    status: Output only. The status of the repo connection request.
  """
    parent = _messages.StringField(1)
    repo = _messages.MessageField('GitLabRepositoryId', 2)
    status = _messages.MessageField('Status', 3)