from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1ColumnSchema(_messages.Message):
    """Representation of a column within a schema. Columns could be nested
  inside other columns.

  Fields:
    column: Required. Name of the column.
    description: Optional. Description of the column. Default value is an
      empty string.
    mode: Optional. A column's mode indicates whether the values in this
      column are required, nullable, etc. Only `NULLABLE`, `REQUIRED` and
      `REPEATED` are supported. Default mode is `NULLABLE`.
    subcolumns: Optional. Schema of sub-columns. A column can have zero or
      more sub-columns.
    type: Required. Type of the column.
  """
    column = _messages.StringField(1)
    description = _messages.StringField(2)
    mode = _messages.StringField(3)
    subcolumns = _messages.MessageField('GoogleCloudDatacatalogV1beta1ColumnSchema', 4, repeated=True)
    type = _messages.StringField(5)