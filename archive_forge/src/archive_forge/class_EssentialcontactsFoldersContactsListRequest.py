from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsFoldersContactsListRequest(_messages.Message):
    """A EssentialcontactsFoldersContactsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of results to return from this
      request. Non-positive values are ignored. The presence of
      `next_page_token` in the response indicates that more results might be
      available. If not specified, the default page_size is 100.
    pageToken: Optional. If present, retrieves the next batch of results from
      the preceding call to this method. `page_token` must be the value of
      `next_page_token` from the previous response. The values of other method
      parameters should be identical to those in the previous call.
    parent: Required. The parent resource name. Format:
      organizations/{organization_id}, folders/{folder_id} or
      projects/{project_id}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)