from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferTransferJobsListRequest(_messages.Message):
    """A StoragetransferTransferJobsListRequest object.

  Fields:
    filter: Required. A list of query parameters specified as JSON text in the
      form of: `{"projectId":"my_project_id",
      "jobNames":["jobid1","jobid2",...],
      "jobStatuses":["status1","status2",...]}` Since `jobNames` and
      `jobStatuses` support multiple values, their values must be specified
      with array notation. `projectId` is required. `jobNames` and
      `jobStatuses` are optional. The valid values for `jobStatuses` are case-
      insensitive: ENABLED, DISABLED, and DELETED.
    pageSize: The list page size. The max allowed value is 256.
    pageToken: The list page token.
  """
    filter = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)