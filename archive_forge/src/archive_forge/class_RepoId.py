from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepoId(_messages.Message):
    """A unique identifier for a Cloud Repo.

  Fields:
    projectRepoId: A combination of a project ID and a repo name.
    uid: A server-assigned, globally unique identifier.
  """
    projectRepoId = _messages.MessageField('ProjectRepoId', 1)
    uid = _messages.StringField(2)