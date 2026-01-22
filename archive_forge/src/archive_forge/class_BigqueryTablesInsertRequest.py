from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryTablesInsertRequest(_messages.Message):
    """A BigqueryTablesInsertRequest object.

  Fields:
    datasetId: Dataset ID of the new table
    projectId: Project ID of the new table
    table: A Table resource to be passed as the request body.
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    table = _messages.MessageField('Table', 3)