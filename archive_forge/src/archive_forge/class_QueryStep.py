from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryStep(_messages.Message):
    """One step of an analytics query. Each query step other than the first
  implicitly takes the output of the previous step as its input. Steps will be
  executed in sequence and will return their results independently (in other
  words, each step will finish at a different time and potentially return a
  different schema).

  Fields:
    alertingQueryStep: A query step that builds an alerting query from
      configuration data.
    chartingQueryStep: A query step that builds a charting query from
      configuration data.
    handleQueryStep: A query step that refers to a step within a previously-
      executed query.
    outputNotRequired: Optional. Set this flag to indicate that you don't
      intend to retrieve the results for this query step. No handle will be
      generated for the step in the QueryDataResponse message.
    sqlQueryStep: A query step containing a SQL query.
  """
    alertingQueryStep = _messages.MessageField('AlertingQueryStep', 1)
    chartingQueryStep = _messages.MessageField('ChartingQueryStep', 2)
    handleQueryStep = _messages.MessageField('HandleQueryStep', 3)
    outputNotRequired = _messages.BooleanField(4)
    sqlQueryStep = _messages.MessageField('SqlQueryStep', 5)