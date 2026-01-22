from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListIdpCredentialsResponse(_messages.Message):
    """Response of the InboundSamlSsoProfilesService.ListIdpCredentials method.

  Fields:
    idpCredentials: The IdpCredentials from the specified
      InboundSamlSsoProfile.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    idpCredentials = _messages.MessageField('IdpCredential', 1, repeated=True)
    nextPageToken = _messages.StringField(2)