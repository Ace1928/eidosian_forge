from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkforceIdentityBasedOAuth2ClientID(_messages.Message):
    """OAuth Client ID depending on the Workforce Identity i.e. either 1p or
  3p,

  Fields:
    firstPartyOauth2ClientId: Output only. First party OAuth Client ID for
      Google Identities.
    thirdPartyOauth2ClientId: Output only. Third party OAuth Client ID for
      External Identity Providers.
  """
    firstPartyOauth2ClientId = _messages.StringField(1)
    thirdPartyOauth2ClientId = _messages.StringField(2)