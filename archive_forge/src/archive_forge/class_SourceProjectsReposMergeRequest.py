from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposMergeRequest(_messages.Message):
    """A SourceProjectsReposMergeRequest object.

  Fields:
    mergeRequest: A MergeRequest resource to be passed as the request body.
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
  """
    mergeRequest = _messages.MessageField('MergeRequest', 1)
    projectId = _messages.StringField(2, required=True)
    repoName = _messages.StringField(3, required=True)