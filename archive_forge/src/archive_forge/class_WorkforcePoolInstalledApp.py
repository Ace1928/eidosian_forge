from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkforcePoolInstalledApp(_messages.Message):
    """Represents a workforce pool installed app. Used to indicate that a
  workforce pool administrator has completed the installation process, thereby
  giving consent for the installed app, i.e. OAuth Client, to access workforce
  pool users' information and resources.

  Enums:
    StateValueValuesEnum: Output only. The state of the workforce pool
      installed app.

  Fields:
    appId: Output only. The UUID of the app that is installed. Current only
      OAuth Client is supported. If the installed app is an OAuth client, this
      field represents the system generated OAuth client ID.
    createTime: Output only. The timestamp when the workforce pool installed
      app was created.
    deleteTime: Output only. The timestamp that the workforce pool installed
      app was soft deleted.
    description: Optional. A user-specified description of the workforce pool
      installed app. Cannot exceed 256 characters.
    displayName: Optional. A user-specified display name of the workforce pool
      installed app Cannot exceed 32 characters.
    expireTime: Output only. Time after which the workforce pool installed app
      will be permanently purged and cannot be recovered.
    name: Immutable. The resource name of the workforce pool installed app.
      Format: `locations/{location}/workforcePools/{workforce_pool}/installedA
      pps/{installed_app}`
    oauthClient: Immutable. The resource name of an OAuth client to be
      installed. Format:
      `projects/{project}/locations/{location}/oauthClients/{oauth_client}`.
    state: Output only. The state of the workforce pool installed app.
    updateTime: Output only. The timestamp for the last update of the
      workforce pool installed app.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the workforce pool installed app.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: The workforce pool installed app is active.
      DELETED: The workforce pool installed app is soft-deleted. Soft-deleted
        workforce pool installed apps are permanently deleted after
        approximately 30 days unless restored via
        UndeleteWorkforcePoolInstalledApp.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2
    appId = _messages.StringField(1)
    createTime = _messages.StringField(2)
    deleteTime = _messages.StringField(3)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
    expireTime = _messages.StringField(6)
    name = _messages.StringField(7)
    oauthClient = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    updateTime = _messages.StringField(10)