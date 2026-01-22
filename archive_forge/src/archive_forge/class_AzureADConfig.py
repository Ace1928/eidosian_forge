from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AzureADConfig(_messages.Message):
    """Configuration for the AzureAD Auth flow.

  Fields:
    clientId: ID for the registered client application that makes
      authentication requests to the Azure AD identity provider.
    clientSecret: Input only. Unencrypted AzureAD client secret will be passed
      to the GKE Hub CLH.
    encryptedClientSecret: Output only. Encrypted AzureAD client secret.
    groupFormat: Optional. Format of the AzureAD groups that the client wants
      for auth.
    kubectlRedirectUri: The redirect URL that kubectl uses for authorization.
    tenant: Kind of Azure AD account to be authenticated. Supported values are
      or for accounts belonging to a specific tenant.
    userClaim: Optional. Claim in the AzureAD ID Token that holds the user
      details.
  """
    clientId = _messages.StringField(1)
    clientSecret = _messages.StringField(2)
    encryptedClientSecret = _messages.BytesField(3)
    groupFormat = _messages.StringField(4)
    kubectlRedirectUri = _messages.StringField(5)
    tenant = _messages.StringField(6)
    userClaim = _messages.StringField(7)