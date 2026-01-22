from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateGitLabConnectedRepositoriesResponse(_messages.Message):
    """Response of BatchCreateGitLabConnectedRepositories RPC method.

  Fields:
    gitlabConnectedRepositories: The GitLab connected repository requests'
      responses.
  """
    gitlabConnectedRepositories = _messages.MessageField('GitLabConnectedRepository', 1, repeated=True)