from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OAuthCredential(_messages.Message):
    """Represents an OAuth token of the account that authorized the Connection,
  and associated metadata.

  Fields:
    oauthTokenSecretVersion: A SecretManager resource containing the OAuth
      token that authorizes the Cloud Build connection. Format:
      `projects/*/secrets/*/versions/*`.
    username: Output only. The username associated to this token.
  """
    oauthTokenSecretVersion = _messages.StringField(1)
    username = _messages.StringField(2)