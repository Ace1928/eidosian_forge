from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRecognizersResponse(_messages.Message):
    """Response message for the ListRecognizers method.

  Fields:
    nextPageToken: A token, which can be sent as page_token to retrieve the
      next page. If this field is omitted, there are no subsequent pages. This
      token expires after 72 hours.
    recognizers: The list of requested Recognizers.
  """
    nextPageToken = _messages.StringField(1)
    recognizers = _messages.MessageField('Recognizer', 2, repeated=True)