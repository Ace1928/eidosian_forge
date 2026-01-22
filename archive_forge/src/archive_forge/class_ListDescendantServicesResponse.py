from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDescendantServicesResponse(_messages.Message):
    """The response message of the `ListDescendantServices` method.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    services: The services the group depends on, directly or indirectly.
  """
    nextPageToken = _messages.StringField(1)
    services = _messages.MessageField('DescendantService', 2, repeated=True)