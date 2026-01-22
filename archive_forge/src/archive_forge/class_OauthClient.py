from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OauthClient(_messages.Message):
    """Represents an oauth client. Used to access Google Cloud resources on
  behave of a user by using OAuth2 Protocol to obtain an access token from
  Google Cloud Platform.

  Enums:
    AllowedGrantTypesValueListEntryValuesEnum:
    ClientTypeValueValuesEnum: Immutable. The type of oauth client. either
      public or private.
    StateValueValuesEnum: Output only. The state of the oauth client.

  Fields:
    allowedGrantTypes: Required. The list of OAuth grant type is allowed for
      the oauth client.
    allowedRedirectUris: Required. The list of redirect uris that is allowed
      to redirect back when authorization process is completed.
    allowedScopes: Required. The list of scopes that the oauth client is
      allowed to request during OAuth flows. The following scopes are
      supported: * `https://www.googleapis.com/auth/cloud-platform`: See,
      edit, configure, and delete your Google Cloud data and see the email
      address for your Google Account. * `openid`: Associate you with your
      personal info on Google Cloud. * `email`: See your Google Cloud Account
      email address.
    clientId: Output only. The system-generated oauth client id.
    clientType: Immutable. The type of oauth client. either public or private.
    description: Optional. A user-specified description of the oauth client.
      Cannot exceed 256 characters.
    disabled: Optional. Whether the oauth client is disabled. You cannot use a
      disabled oauth client for login.
    displayName: Optional. A user-specified display name of the oauth client.
      Cannot exceed 32 characters.
    expireTime: Output only. Time after which the oauth client will be
      permanently purged and cannot be recovered.
    name: Immutable. The resource name of the oauth client. Format:`projects/{
      project}/locations/{location}/oauthClients/{oauth_client}`.
    state: Output only. The state of the oauth client.
  """

    class AllowedGrantTypesValueListEntryValuesEnum(_messages.Enum):
        """AllowedGrantTypesValueListEntryValuesEnum enum type.

    Values:
      GRANT_TYPE_UNSPECIFIED: should not be used
      AUTHORIZATION_CODE_GRANT: authorization code grant
      REFRESH_TOKEN_GRANT: refresh token grant
    """
        GRANT_TYPE_UNSPECIFIED = 0
        AUTHORIZATION_CODE_GRANT = 1
        REFRESH_TOKEN_GRANT = 2

    class ClientTypeValueValuesEnum(_messages.Enum):
        """Immutable. The type of oauth client. either public or private.

    Values:
      CLIENT_TYPE_UNSPECIFIED: should not be used
      PUBLIC_CLIENT: public client has no secret
      CONFIDENTIAL_CLIENT: private client
    """
        CLIENT_TYPE_UNSPECIFIED = 0
        PUBLIC_CLIENT = 1
        CONFIDENTIAL_CLIENT = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the oauth client.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: The oauth client is active.
      DELETED: The oauth client is soft-deleted. Soft-deleted oauth client is
        permanently deleted after approximately 30 days unless restored via
        UndeleteOauthClient.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2
    allowedGrantTypes = _messages.EnumField('AllowedGrantTypesValueListEntryValuesEnum', 1, repeated=True)
    allowedRedirectUris = _messages.StringField(2, repeated=True)
    allowedScopes = _messages.StringField(3, repeated=True)
    clientId = _messages.StringField(4)
    clientType = _messages.EnumField('ClientTypeValueValuesEnum', 5)
    description = _messages.StringField(6)
    disabled = _messages.BooleanField(7)
    displayName = _messages.StringField(8)
    expireTime = _messages.StringField(9)
    name = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)