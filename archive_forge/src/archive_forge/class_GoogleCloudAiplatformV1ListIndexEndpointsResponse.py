from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListIndexEndpointsResponse(_messages.Message):
    """Response message for IndexEndpointService.ListIndexEndpoints.

  Fields:
    indexEndpoints: List of IndexEndpoints in the requested page.
    nextPageToken: A token to retrieve next page of results. Pass to
      ListIndexEndpointsRequest.page_token to obtain that page.
  """
    indexEndpoints = _messages.MessageField('GoogleCloudAiplatformV1IndexEndpoint', 1, repeated=True)
    nextPageToken = _messages.StringField(2)