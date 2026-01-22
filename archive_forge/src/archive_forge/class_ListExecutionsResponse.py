from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListExecutionsResponse(_messages.Message):
    """A ListExecutionsResponse object.

  Fields:
    executions: Executions. Always set.
    nextPageToken: A continuation token to resume the query at the next item.
      Will only be set if there are more Executions to fetch.
  """
    executions = _messages.MessageField('Execution', 1, repeated=True)
    nextPageToken = _messages.StringField(2)