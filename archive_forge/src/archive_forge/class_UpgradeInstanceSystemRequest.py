from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeInstanceSystemRequest(_messages.Message):
    """Request for upgrading a notebook instance from within the VM

  Fields:
    vmId: Required. The VM hardware token for authenticating the VM.
      https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity
  """
    vmId = _messages.StringField(1)