from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GCPSecretManagerCertificateConfig(_messages.Message):
    """GCPSecretManagerCertificateConfig configures a secret from [Google
  Secret Manager](https://cloud.google.com/secret-manager).

  Fields:
    secretUri: Secret URI, in the form
      "projects/$PROJECT_ID/secrets/$SECRET_NAME/versions/$VERSION". Version
      can be fixed (e.g. "2") or "latest"
  """
    secretUri = _messages.StringField(1)