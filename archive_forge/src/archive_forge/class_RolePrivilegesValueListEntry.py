from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolePrivilegesValueListEntry(_messages.Message):
    """A RolePrivilegesValueListEntry object.

    Fields:
      privilegeName: The name of the privilege.
      serviceId: The obfuscated ID of the service this privilege is for. This
        value is returned with Privileges.list().
    """
    privilegeName = _messages.StringField(1)
    serviceId = _messages.StringField(2)