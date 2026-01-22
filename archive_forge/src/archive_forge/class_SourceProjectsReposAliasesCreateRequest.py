from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposAliasesCreateRequest(_messages.Message):
    """A SourceProjectsReposAliasesCreateRequest object.

  Fields:
    alias: A Alias resource to be passed as the request body.
    projectId: The ID of the project.
    repoId_uid: A server-assigned, globally unique identifier.
    repoName: The name of the repo. Leave empty for the default repo.
  """
    alias = _messages.MessageField('Alias', 1)
    projectId = _messages.StringField(2, required=True)
    repoId_uid = _messages.StringField(3)
    repoName = _messages.StringField(4, required=True)