from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTargetsResponse(_messages.Message):
    """The response object from `ListTargets`.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    targets: The `Target` objects.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    targets = _messages.MessageField('Target', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)