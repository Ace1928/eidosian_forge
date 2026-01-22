from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryRoleAssignmentsDeleteRequest(_messages.Message):
    """A DirectoryRoleAssignmentsDeleteRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
    roleAssignmentId: Immutable ID of the role assignment.
  """
    customer = _messages.StringField(1, required=True)
    roleAssignmentId = _messages.StringField(2, required=True)