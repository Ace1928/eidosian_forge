from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityRuleRowConditionExpectation(_messages.Message):
    """Evaluates whether each row passes the specified condition.The SQL
  expression needs to use BigQuery standard SQL syntax and should produce a
  boolean value per row as the result.Example: col1 >= 0 AND col2 < 10

  Fields:
    sqlExpression: Optional. The SQL expression.
  """
    sqlExpression = _messages.StringField(1)