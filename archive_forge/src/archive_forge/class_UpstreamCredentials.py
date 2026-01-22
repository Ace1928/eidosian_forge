from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpstreamCredentials(_messages.Message):
    """The credentials to access the remote repository.

  Fields:
    usernamePasswordCredentials: Use username and password to access the
      remote repository.
  """
    usernamePasswordCredentials = _messages.MessageField('UsernamePasswordCredentials', 1)