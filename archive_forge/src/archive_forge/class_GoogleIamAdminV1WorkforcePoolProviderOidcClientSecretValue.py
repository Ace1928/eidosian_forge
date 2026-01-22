from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamAdminV1WorkforcePoolProviderOidcClientSecretValue(_messages.Message):
    """Representation of the value of the client secret.

  Fields:
    plainText: Input only. The plain text of the client secret value. For
      security reasons, this field is only used for input and will never be
      populated in any response.
    thumbprint: Output only. A thumbprint to represent the current client
      secret value.
  """
    plainText = _messages.StringField(1)
    thumbprint = _messages.StringField(2)