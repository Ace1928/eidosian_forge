from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDatasetsResponse(_messages.Message):
    """Response message for ListDatasets.

  Fields:
    datasets: The datasets read.
    nextPageToken: A token to retrieve next page of results. Pass this token
      to the page_token field in the ListDatasetsRequest to obtain the
      corresponding page.
  """
    datasets = _messages.MessageField('Dataset', 1, repeated=True)
    nextPageToken = _messages.StringField(2)