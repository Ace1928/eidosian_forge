from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryTablesDeleteRequest(_messages.Message):
    """A BigqueryTablesDeleteRequest object.

  Fields:
    datasetId: Dataset ID of the table to delete
    projectId: Project ID of the table to delete
    tableId: Table ID of the table to delete
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    tableId = _messages.StringField(3, required=True)