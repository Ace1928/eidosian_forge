from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSparkApplicationSqlQueryResponse(_messages.Message):
    """Details of a query for a Spark Application

  Fields:
    executionData: SQL Execution Data
  """
    executionData = _messages.MessageField('SqlExecutionUiData', 1)