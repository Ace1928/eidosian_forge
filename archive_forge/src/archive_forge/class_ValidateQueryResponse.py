from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateQueryResponse(_messages.Message):
    """The response data from ValidateQuery.

  Fields:
    validateResult: The operation does basic syntactic validation on all steps
      and will return an error if an issue is found. Only the first query step
      is validated through BigQuery, however, and only if it's a SqlQueryStep.
      If the first step is not SQL, this field will be empty.
  """
    validateResult = _messages.MessageField('QueryResults', 1)