from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SearchProtectedResourcesResponse(_messages.Message):
    """Response message for KeyTrackingService.SearchProtectedResources.

  Fields:
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    protectedResources: Protected resources for this page.
  """
    nextPageToken = _messages.StringField(1)
    protectedResources = _messages.MessageField('ProtectedResource', 2, repeated=True)