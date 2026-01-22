from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransformColumn(_messages.Message):
    """Information about a single transform column.

  Fields:
    name: Output only. Name of the column.
    transformSql: Output only. The SQL expression used in the column
      transform.
    type: Output only. Data type of the column after the transform.
  """
    name = _messages.StringField(1)
    transformSql = _messages.StringField(2)
    type = _messages.MessageField('StandardSqlDataType', 3)