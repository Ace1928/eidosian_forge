from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationCodeLink(_messages.Message):
    """This configuration captures the details required to render an
  authorization link for the OAuth Authorization Code Flow.

  Fields:
    clientId: The client ID assigned to the Google Cloud Connectors OAuth app
      for the connector data source.
    enablePkce: Whether to enable PKCE for the auth code flow.
    scopes: The scopes for which the user will authorize Google Cloud
      Connectors on the connector data source.
    uri: The base URI the user must click to trigger the authorization code
      login flow.
  """
    clientId = _messages.StringField(1)
    enablePkce = _messages.BooleanField(2)
    scopes = _messages.StringField(3, repeated=True)
    uri = _messages.StringField(4)