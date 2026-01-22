from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsernamePasswordCredentials(_messages.Message):
    """Username and password credentials.

  Fields:
    passwordSecretVersion: The Secret Manager key version that holds the
      password to access the remote repository. Must be in the format of
      `projects/{project}/secrets/{secret}/versions/{version}`.
    username: The username to access the remote repository.
  """
    passwordSecretVersion = _messages.StringField(1)
    username = _messages.StringField(2)