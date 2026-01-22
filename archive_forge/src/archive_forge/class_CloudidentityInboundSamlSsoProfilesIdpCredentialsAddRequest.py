from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityInboundSamlSsoProfilesIdpCredentialsAddRequest(_messages.Message):
    """A CloudidentityInboundSamlSsoProfilesIdpCredentialsAddRequest object.

  Fields:
    addIdpCredentialRequest: A AddIdpCredentialRequest resource to be passed
      as the request body.
    parent: Required. The InboundSamlSsoProfile that owns the IdpCredential.
      Format: `inboundSamlSsoProfiles/{sso_profile_id}`
  """
    addIdpCredentialRequest = _messages.MessageField('AddIdpCredentialRequest', 1)
    parent = _messages.StringField(2, required=True)