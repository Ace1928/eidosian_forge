from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRolesResponse(_messages.Message):
    """The response containing the roles defined under a resource.

  Fields:
    nextPageToken: To retrieve the next page of results, set
      `ListRolesRequest.page_token` to this value.
    roles: The Roles defined on this resource.
  """
    nextPageToken = _messages.StringField(1)
    roles = _messages.MessageField('Role', 2, repeated=True)