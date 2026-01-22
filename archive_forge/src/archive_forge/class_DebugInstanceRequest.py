from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DebugInstanceRequest(_messages.Message):
    """Request message for Instances.DebugInstance.

  Fields:
    sshKey: Public SSH key to add to the instance. Examples: [USERNAME]:ssh-
      rsa [KEY_VALUE] [USERNAME] [USERNAME]:ssh-rsa [KEY_VALUE] google-ssh
      {"userName":"[USERNAME]","expireOn":"[EXPIRE_TIME]"}For more
      information, see Adding and Removing SSH Keys
      (https://cloud.google.com/compute/docs/instances/adding-removing-ssh-
      keys).
  """
    sshKey = _messages.StringField(1)