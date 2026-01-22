from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTagValuesResponse(_messages.Message):
    """The ListTagValues response.

  Fields:
    nextPageToken: A pagination token returned from a previous call to
      `ListTagValues` that indicates from where listing should continue. This
      is currently not used, but the server may at any point start supplying a
      valid token.
    tagValues: A possibly paginated list of TagValues that are direct
      descendants of the specified parent TagKey.
  """
    nextPageToken = _messages.StringField(1)
    tagValues = _messages.MessageField('TagValue', 2, repeated=True)