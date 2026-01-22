from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryTablesListRequest(_messages.Message):
    """A BigqueryTablesListRequest object.

  Fields:
    datasetId: Dataset ID of the tables to list
    maxResults: Maximum number of results to return
    pageToken: Page token, returned by a previous call, to request the next
      page of results
    projectId: Project ID of the tables to list
  """
    datasetId = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)