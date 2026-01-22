from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryTabledataInsertAllRequest(_messages.Message):
    """A BigqueryTabledataInsertAllRequest object.

  Fields:
    datasetId: Dataset ID of the destination table.
    projectId: Project ID of the destination table.
    tableDataInsertAllRequest: A TableDataInsertAllRequest resource to be
      passed as the request body.
    tableId: Table ID of the destination table.
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    tableDataInsertAllRequest = _messages.MessageField('TableDataInsertAllRequest', 3)
    tableId = _messages.StringField(4, required=True)