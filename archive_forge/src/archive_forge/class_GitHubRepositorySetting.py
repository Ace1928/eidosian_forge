from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitHubRepositorySetting(_messages.Message):
    """Represents a GitHub repository setting.

  Fields:
    name: Name of the repository.
    owner: GitHub user or organization name.
  """
    name = _messages.StringField(1)
    owner = _messages.StringField(2)