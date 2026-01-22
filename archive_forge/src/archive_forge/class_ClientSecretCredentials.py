from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientSecretCredentials(_messages.Message):
    """Message describing Azure Credentials using tenant ID, client ID and
  secret.

  Fields:
    clientId: Azure client ID.
    clientSecret: Input only. Azure client secret.
    tenantId: Azure tenant ID.
  """
    clientId = _messages.StringField(1)
    clientSecret = _messages.StringField(2)
    tenantId = _messages.StringField(3)