from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepoSource(_messages.Message):
    """RepoSource describes the location of the source in a Google Cloud Source
  Repository.

  Fields:
    branchName: Name of the branch to build.
    commitSha: Explicit commit SHA to build.
    projectId: ID of the project that owns the repo.
    repoName: Name of the repo.
    tagName: Name of the tag to build.
  """
    branchName = _messages.StringField(1)
    commitSha = _messages.StringField(2)
    projectId = _messages.StringField(3)
    repoName = _messages.StringField(4)
    tagName = _messages.StringField(5)