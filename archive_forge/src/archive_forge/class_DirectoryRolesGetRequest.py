from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryRolesGetRequest(_messages.Message):
    """A DirectoryRolesGetRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
    roleId: Immutable ID of the role.
  """
    customer = _messages.StringField(1, required=True)
    roleId = _messages.StringField(2, required=True)