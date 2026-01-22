from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListExclusionsResponse(_messages.Message):
    """Result returned from ListExclusions.

  Fields:
    exclusions: A list of exclusions.
    nextPageToken: If there might be more results than appear in this
      response, then nextPageToken is included. To get the next set of
      results, call the same method again using the value of nextPageToken as
      pageToken.
  """
    exclusions = _messages.MessageField('LogExclusion', 1, repeated=True)
    nextPageToken = _messages.StringField(2)