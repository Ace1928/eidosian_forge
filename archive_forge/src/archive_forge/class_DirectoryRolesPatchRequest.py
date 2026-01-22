from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryRolesPatchRequest(_messages.Message):
    """A DirectoryRolesPatchRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
    role: A Role resource to be passed as the request body.
    roleId: Immutable ID of the role.
  """
    customer = _messages.StringField(1, required=True)
    role = _messages.MessageField('Role', 2)
    roleId = _messages.StringField(3, required=True)