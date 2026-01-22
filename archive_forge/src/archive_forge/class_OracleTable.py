from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleTable(_messages.Message):
    """Oracle table.

  Fields:
    oracleColumns: Oracle columns in the schema. When unspecified as part of
      include/exclude objects, includes/excludes everything.
    table: Table name.
  """
    oracleColumns = _messages.MessageField('OracleColumn', 1, repeated=True)
    table = _messages.StringField(2)