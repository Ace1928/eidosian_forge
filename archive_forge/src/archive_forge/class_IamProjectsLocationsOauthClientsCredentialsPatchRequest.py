from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsOauthClientsCredentialsPatchRequest(_messages.Message):
    """A IamProjectsLocationsOauthClientsCredentialsPatchRequest object.

  Fields:
    name: Immutable. The resource name of the oauth client credential. Format:
      `projects/{project}/locations/{location}/oauthClients/{oauth_client}/cre
      dentials/{credential}`
    oauthClientCredential: A OauthClientCredential resource to be passed as
      the request body.
    updateMask: Required. The list of fields to update.
  """
    name = _messages.StringField(1, required=True)
    oauthClientCredential = _messages.MessageField('OauthClientCredential', 2)
    updateMask = _messages.StringField(3)