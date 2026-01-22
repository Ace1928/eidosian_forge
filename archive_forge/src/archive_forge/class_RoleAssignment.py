from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoleAssignment(_messages.Message):
    """JSON template for roleAssignment resource in Directory API.

  Fields:
    assignedTo: The unique ID of the user this role is assigned to.
    etag: ETag of the resource.
    kind: The type of the API resource. This is always
      admin#directory#roleAssignment.
    orgUnitId: If the role is restricted to an organization unit, this
      contains the ID for the organization unit the exercise of this role is
      restricted to.
    roleAssignmentId: ID of this roleAssignment.
    roleId: The ID of the role that is assigned.
    scopeType: The scope in which this role is assigned. Possible values are:
      - CUSTOMER - ORG_UNIT
  """
    assignedTo = _messages.StringField(1)
    etag = _messages.StringField(2)
    kind = _messages.StringField(3, default=u'admin#directory#roleAssignment')
    orgUnitId = _messages.StringField(4)
    roleAssignmentId = _messages.IntegerField(5)
    roleId = _messages.IntegerField(6)
    scopeType = _messages.StringField(7)