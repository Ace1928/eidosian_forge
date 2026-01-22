from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceConfig(_messages.Message):
    """IdentityServiceConfig is configuration for Identity Service which allows
  customers to use external identity providers with the K8S API

  Fields:
    enabled: Whether to enable the Identity Service component
  """
    enabled = _messages.BooleanField(1)