from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IapCredential(_messages.Message):
    """Describes authentication configuration for Identity-Aware-Proxy (IAP).

  Fields:
    iapTestServiceAccountInfo: Authentication configuration when Web-Security-
      Scanner service account is added in Identity-Aware-Proxy (IAP) access
      policies.
  """
    iapTestServiceAccountInfo = _messages.MessageField('IapTestServiceAccountInfo', 1)