from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoleAssignments(_messages.Message):
    """JSON response template for List roleAssignments operation in Directory

  API.

  Fields:
    etag: ETag of the resource.
    items: A list of RoleAssignment resources.
    kind: The type of the API resource. This is always
      admin#directory#roleAssignments.
    nextPageToken: A string attribute.
  """
    etag = _messages.StringField(1)
    items = _messages.MessageField('RoleAssignment', 2, repeated=True)
    kind = _messages.StringField(3, default=u'admin#directory#roleAssignments')
    nextPageToken = _messages.StringField(4)