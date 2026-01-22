from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitHubEnterpriseSecrets(_messages.Message):
    """GitHubEnterpriseSecrets represents the names of all necessary secrets in
  Secret Manager for a GitHub Enterprise server. Format is:
  projects//secrets/.

  Fields:
    oauthClientIdName: The resource name for the OAuth client ID secret in
      Secret Manager.
    oauthClientIdVersionName: The resource name for the OAuth client ID secret
      version in Secret Manager.
    oauthSecretName: The resource name for the OAuth secret in Secret Manager.
    oauthSecretVersionName: The resource name for the OAuth secret secret
      version in Secret Manager.
    privateKeyName: The resource name for the private key secret.
    privateKeyVersionName: The resource name for the private key secret
      version.
    webhookSecretName: The resource name for the webhook secret in Secret
      Manager.
    webhookSecretVersionName: The resource name for the webhook secret secret
      version in Secret Manager.
  """
    oauthClientIdName = _messages.StringField(1)
    oauthClientIdVersionName = _messages.StringField(2)
    oauthSecretName = _messages.StringField(3)
    oauthSecretVersionName = _messages.StringField(4)
    privateKeyName = _messages.StringField(5)
    privateKeyVersionName = _messages.StringField(6)
    webhookSecretName = _messages.StringField(7)
    webhookSecretVersionName = _messages.StringField(8)