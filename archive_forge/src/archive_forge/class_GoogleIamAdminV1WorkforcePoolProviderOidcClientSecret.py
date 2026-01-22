from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamAdminV1WorkforcePoolProviderOidcClientSecret(_messages.Message):
    """Representation of a client secret configured for the OIDC provider.

  Fields:
    value: The value of the client secret.
  """
    value = _messages.MessageField('GoogleIamAdminV1WorkforcePoolProviderOidcClientSecretValue', 1)