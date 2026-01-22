from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListDataCollectorsResponse(_messages.Message):
    """Response for ListDataCollectors.

  Fields:
    dataCollectors: Data collectors in the specified organization.
    nextPageToken: Page token that you can include in a ListDataCollectors
      request to retrieve the next page. If omitted, no subsequent pages
      exist.
  """
    dataCollectors = _messages.MessageField('GoogleCloudApigeeV1DataCollector', 1, repeated=True)
    nextPageToken = _messages.StringField(2)