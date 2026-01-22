from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OAuthSettings(_messages.Message):
    """Configuration for OAuth login&consent flow behavior as well as for OAuth
  Credentials.

  Fields:
    loginHint: Domain hint to send as hd=? parameter in OAuth request flow.
      Enables redirect to primary IDP by skipping Google's login screen.
      https://developers.google.com/identity/protocols/OpenIDConnect#hd-param
      Note: IAP does not verify that the id token's hd claim matches this
      value since access behavior is managed by IAM policies.
    programmaticClients: List of client ids allowed to use IAP
      programmatically.
  """
    loginHint = _messages.StringField(1)
    programmaticClients = _messages.StringField(2, repeated=True)