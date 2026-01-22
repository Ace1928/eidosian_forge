from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudtraceProjectsTraceSinksListRequest(_messages.Message):
    """A CloudtraceProjectsTraceSinksListRequest object.

  Fields:
    pageSize: Optional. The maximum number of results to return from this
      request. Non-positive values are ignored. The presence of
      `next_page_token` in the response indicates that more results might be
      available.
    pageToken: Optional. If present, then retrieve the next batch of results
      from the preceding call to this method. `page_token` must be the value
      of `next_page_token` from the previous response. The values of other
      method parameters should be identical to those in the previous call.
    parent: Required. The parent resource whose sinks are to be listed
      (currently only project parent resources are supported):
      "projects/[PROJECT_ID]"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)