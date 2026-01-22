from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Automatic(_messages.Message):
    """A replication policy that replicates the Secret payload without any
  restrictions.

  Fields:
    customerManagedEncryption: Optional. The customer-managed encryption
      configuration of the Secret. If no configuration is provided, Google-
      managed default encryption is used. Updates to the Secret encryption
      configuration only apply to SecretVersions added afterwards. They do not
      apply retroactively to existing SecretVersions.
  """
    customerManagedEncryption = _messages.MessageField('CustomerManagedEncryption', 1)