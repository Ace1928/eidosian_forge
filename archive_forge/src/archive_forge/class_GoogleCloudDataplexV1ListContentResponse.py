from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListContentResponse(_messages.Message):
    """List content response.

  Fields:
    content: Content under the given parent lake.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    content = _messages.MessageField('GoogleCloudDataplexV1Content', 1, repeated=True)
    nextPageToken = _messages.StringField(2)