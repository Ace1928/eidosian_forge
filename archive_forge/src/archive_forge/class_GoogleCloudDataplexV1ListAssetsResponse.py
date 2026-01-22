from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListAssetsResponse(_messages.Message):
    """List assets response.

  Fields:
    assets: Asset under the given parent zone.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    assets = _messages.MessageField('GoogleCloudDataplexV1Asset', 1, repeated=True)
    nextPageToken = _messages.StringField(2)