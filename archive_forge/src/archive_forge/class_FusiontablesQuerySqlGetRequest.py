from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesQuerySqlGetRequest(_messages.Message):
    """A FusiontablesQuerySqlGetRequest object.

  Fields:
    hdrs: Should column names be included (in the first row)?. Default is
      true.
    sql: An SQL SELECT/SHOW/DESCRIBE statement.
    typed: Should typed values be returned in the (JSON) response -- numbers
      for numeric values and parsed geometries for KML values? Default is
      true.
  """
    hdrs = _messages.BooleanField(1)
    sql = _messages.StringField(2, required=True)
    typed = _messages.BooleanField(3)