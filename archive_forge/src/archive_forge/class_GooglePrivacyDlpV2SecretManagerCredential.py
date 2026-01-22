from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2SecretManagerCredential(_messages.Message):
    """A credential consisting of a username and password, where the password
  is stored in a Secret Manager resource. Note: Secret Manager [charges
  apply](https://cloud.google.com/secret-manager/pricing).

  Fields:
    passwordSecretVersionName: Required. The name of the Secret Manager
      resource that stores the password, in the form "projects/project-
      id/secrets/secret-name/versions/version".
    username: Required. The username.
  """
    passwordSecretVersionName = _messages.StringField(1)
    username = _messages.StringField(2)