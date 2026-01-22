from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSSHKeysResponse(_messages.Message):
    """Message for response of ListSSHKeys.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    sshKeys: The SSH keys registered in the project.
  """
    nextPageToken = _messages.StringField(1)
    sshKeys = _messages.MessageField('SSHKey', 2, repeated=True)