from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryAssetsResponse(_messages.Message):
    """QueryAssets response.

  Fields:
    done: The query response, which can be either an `error` or a valid
      `response`. If `done` == `false` and the query result is being saved in
      a output, the output_config field will be set. If `done` == `true`,
      exactly one of `error`, `query_result` or `output_config` will be set.
    error: Error status.
    jobReference: Reference to a query job.
    outputConfig: Output configuration which indicates instead of being
      returned in API response on the fly, the query result will be saved in a
      specific output.
    queryResult: Result of the query.
  """
    done = _messages.BooleanField(1)
    error = _messages.MessageField('Status', 2)
    jobReference = _messages.StringField(3)
    outputConfig = _messages.MessageField('QueryAssetsOutputConfig', 4)
    queryResult = _messages.MessageField('QueryResult', 5)