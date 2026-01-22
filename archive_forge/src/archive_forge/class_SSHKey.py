from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SSHKey(_messages.Message):
    """An SSH key, used for authorizing with the interactive serial console
  feature.

  Fields:
    name: Output only. The name of this SSH key. Currently, the only valid
      value for the location is "global".
    publicKey: The public SSH key. This must be in OpenSSH .authorized_keys
      format.
  """
    name = _messages.StringField(1)
    publicKey = _messages.StringField(2)