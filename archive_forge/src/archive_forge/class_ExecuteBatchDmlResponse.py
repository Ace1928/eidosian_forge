from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecuteBatchDmlResponse(_messages.Message):
    """The response for ExecuteBatchDml. Contains a list of ResultSet messages,
  one for each DML statement that has successfully executed, in the same order
  as the statements in the request. If a statement fails, the status in the
  response body identifies the cause of the failure. To check for DML
  statements that failed, use the following approach: 1. Check the status in
  the response message. The google.rpc.Code enum value `OK` indicates that all
  statements were executed successfully. 2. If the status was not `OK`, check
  the number of result sets in the response. If the response contains `N`
  ResultSet messages, then statement `N+1` in the request failed. Example 1: *
  Request: 5 DML statements, all executed successfully. * Response: 5
  ResultSet messages, with the status `OK`. Example 2: * Request: 5 DML
  statements. The third statement has a syntax error. * Response: 2 ResultSet
  messages, and a syntax error (`INVALID_ARGUMENT`) status. The number of
  ResultSet messages indicates that the third statement failed, and the fourth
  and fifth statements were not executed.

  Fields:
    resultSets: One ResultSet for each statement in the request that ran
      successfully, in the same order as the statements in the request. Each
      ResultSet does not contain any rows. The ResultSetStats in each
      ResultSet contain the number of rows modified by the statement. Only the
      first ResultSet in the response contains valid ResultSetMetadata.
    status: If all DML statements are executed successfully, the status is
      `OK`. Otherwise, the error status of the first failed statement.
  """
    resultSets = _messages.MessageField('ResultSet', 1, repeated=True)
    status = _messages.MessageField('Status', 2)