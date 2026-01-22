from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsloginUsersImportSshPublicKeyRequest(_messages.Message):
    """A OsloginUsersImportSshPublicKeyRequest object.

  Enums:
    ViewValueValuesEnum: The view configures whether to retrieve security keys
      information.

  Fields:
    parent: The unique ID for the user in format `users/{user}`.
    projectId: The project ID of the Google Cloud Platform project.
    regions: Optional. The regions to which to assert that the key was
      written. If unspecified, defaults to all regions. Regions are listed at
      https://cloud.google.com/about/locations#region.
    sshPublicKey: A SshPublicKey resource to be passed as the request body.
    view: The view configures whether to retrieve security keys information.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view configures whether to retrieve security keys information.

    Values:
      LOGIN_PROFILE_VIEW_UNSPECIFIED: The default login profile view. The API
        defaults to the BASIC view.
      BASIC: Includes POSIX and SSH key information.
      SECURITY_KEY: Include security key information for the user.
    """
        LOGIN_PROFILE_VIEW_UNSPECIFIED = 0
        BASIC = 1
        SECURITY_KEY = 2
    parent = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2)
    regions = _messages.StringField(3, repeated=True)
    sshPublicKey = _messages.MessageField('SshPublicKey', 4)
    view = _messages.EnumField('ViewValueValuesEnum', 5)