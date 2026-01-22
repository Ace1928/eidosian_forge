from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplicationsValueListEntry(_messages.Message):
    """A ApplicationsValueListEntry object.

    Fields:
      displayName: Display name of application
      packageName: Package name of application
      permission: List of Permissions for application
      versionCode: Version code of application
      versionName: Version name of application
    """
    displayName = _messages.StringField(1)
    packageName = _messages.StringField(2)
    permission = _messages.StringField(3, repeated=True)
    versionCode = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    versionName = _messages.StringField(5)