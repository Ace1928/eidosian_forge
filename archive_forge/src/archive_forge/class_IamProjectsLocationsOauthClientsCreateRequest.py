from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsOauthClientsCreateRequest(_messages.Message):
    """A IamProjectsLocationsOauthClientsCreateRequest object.

  Fields:
    oauthClient: A OauthClient resource to be passed as the request body.
    oauthClientId: Required. The ID to use for the oauth client, which becomes
      the final component of the resource name. This value should be a string
      of 6 to 63 lowercase letters, digits, or hyphens. It must start with a
      letter, and cannot have a trailing hyphen. The prefix `gcp-` is reserved
      for use by Google, and may not be specified.
    parent: Required. The parent resource to create the oauth client in. The
      only supported location is `global`.
  """
    oauthClient = _messages.MessageField('OauthClient', 1)
    oauthClientId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)