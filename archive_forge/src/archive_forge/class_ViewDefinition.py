from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ViewDefinition(_messages.Message):
    """A ViewDefinition object.

  Fields:
    query: [Required] A query that BigQuery executes when the view is
      referenced.
    useLegacySql: [Experimental] Specifies whether to use BigQuery's legacy
      SQL for this view. The default value is true. If set to false, the view
      will use BigQuery's standard SQL: https://cloud.google.com/bigquery/sql-
      reference/ Queries and views that reference this view must use the same
      flag value.
    userDefinedFunctionResources: [Experimental] Describes user-defined
      function resources used in the query.
  """
    query = _messages.StringField(1)
    useLegacySql = _messages.BooleanField(2)
    userDefinedFunctionResources = _messages.MessageField('UserDefinedFunctionResource', 3, repeated=True)