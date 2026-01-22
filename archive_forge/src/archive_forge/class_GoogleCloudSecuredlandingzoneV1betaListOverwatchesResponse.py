from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuredlandingzoneV1betaListOverwatchesResponse(_messages.Message):
    """The response message for ListOverwatch.

  Fields:
    nextPageToken: A pagination token to retrieve the next page of results, or
      empty if there are no more results.
    overwatches: List of Overwatch resources under the specified parent in the
      request.
  """
    nextPageToken = _messages.StringField(1)
    overwatches = _messages.MessageField('GoogleCloudSecuredlandingzoneV1betaOverwatch', 2, repeated=True)