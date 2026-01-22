from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndexAdvice(_messages.Message):
    """Recommendation to add new indexes to run queries more efficiently.

  Fields:
    ddl: Optional. DDL statements to add new indexes that will improve the
      query.
    improvementFactor: Optional. Estimated latency improvement factor. For
      example if the query currently takes 500 ms to run and the estimated
      latency with new indexes is 100 ms this field will be 5.
  """
    ddl = _messages.StringField(1, repeated=True)
    improvementFactor = _messages.FloatField(2)