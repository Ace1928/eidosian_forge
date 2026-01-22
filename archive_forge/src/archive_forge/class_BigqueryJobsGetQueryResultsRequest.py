from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryJobsGetQueryResultsRequest(_messages.Message):
    """A BigqueryJobsGetQueryResultsRequest object.

  Fields:
    jobId: [Required] Job ID of the query job
    maxResults: Maximum number of results to read
    pageToken: Page token, returned by a previous call, to request the next
      page of results
    projectId: [Required] Project ID of the query job
    startIndex: Zero-based index of the starting row
    timeoutMs: How long to wait for the query to complete, in milliseconds,
      before returning. Default is 10 seconds. If the timeout passes before
      the job completes, the 'jobComplete' field in the response will be false
  """
    jobId = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    startIndex = _messages.IntegerField(5, variant=_messages.Variant.UINT64)
    timeoutMs = _messages.IntegerField(6, variant=_messages.Variant.UINT32)