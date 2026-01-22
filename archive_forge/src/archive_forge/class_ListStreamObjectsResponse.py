from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListStreamObjectsResponse(_messages.Message):
    """Response containing the objects for a stream.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page.
    streamObjects: List of stream objects.
  """
    nextPageToken = _messages.StringField(1)
    streamObjects = _messages.MessageField('StreamObject', 2, repeated=True)