from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsOauthClientsCredentialsCreateRequest(_messages.Message):
    """A IamProjectsLocationsOauthClientsCredentialsCreateRequest object.

  Fields:
    oauthClientCredential: A OauthClientCredential resource to be passed as
      the request body.
    oauthClientCredentialId: Required. The ID to use for the oauth client
      credential, which becomes the final component of the resource name. This
      value should be 4-32 characters, and may contain the characters
      [a-z0-9-]. The prefix `gcp-` is reserved for use by Google, and may not
      be specified.
    parent: Required. The parent resource to create the oauth client
      Credential in.
  """
    oauthClientCredential = _messages.MessageField('OauthClientCredential', 1)
    oauthClientCredentialId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)