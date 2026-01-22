from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListAsyncQueriesResponse(_messages.Message):
    """The response for ListAsyncQueries.

  Fields:
    queries: The asynchronous queries belong to requested resource name.
  """
    queries = _messages.MessageField('GoogleCloudApigeeV1AsyncQuery', 1, repeated=True)