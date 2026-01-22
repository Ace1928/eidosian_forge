from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityInboundSamlSsoProfilesIdpCredentialsListRequest(_messages.Message):
    """A CloudidentityInboundSamlSsoProfilesIdpCredentialsListRequest object.

  Fields:
    pageSize: The maximum number of `IdpCredential`s to return. The service
      may return fewer than this value.
    pageToken: A page token, received from a previous `ListIdpCredentials`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListIdpCredentials` must match the call
      that provided the page token.
    parent: Required. The parent, which owns this collection of
      `IdpCredential`s. Format: `inboundSamlSsoProfiles/{sso_profile_id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)