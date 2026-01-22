from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableReference(_messages.Message):
    """A TableReference object.

  Fields:
    datasetId: [Required] The ID of the dataset containing this table.
    projectId: [Required] The ID of the project containing this table.
    tableId: [Required] The ID of the table. The ID must contain only letters
      (a-z, A-Z), numbers (0-9), or underscores (_). The maximum length is
      1,024 characters.
  """
    datasetId = _messages.StringField(1)
    projectId = _messages.StringField(2)
    tableId = _messages.StringField(3)