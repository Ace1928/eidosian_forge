from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAdaptiveMtDatasetsResponse(_messages.Message):
    """A list of AdaptiveMtDatasets.

  Fields:
    adaptiveMtDatasets: Output only. A list of Adaptive MT datasets.
    nextPageToken: Optional. A token to retrieve a page of results. Pass this
      value in the [ListAdaptiveMtDatasetsRequest.page_token] field in the
      subsequent call to `ListAdaptiveMtDatasets` method to retrieve the next
      page of results.
  """
    adaptiveMtDatasets = _messages.MessageField('AdaptiveMtDataset', 1, repeated=True)
    nextPageToken = _messages.StringField(2)