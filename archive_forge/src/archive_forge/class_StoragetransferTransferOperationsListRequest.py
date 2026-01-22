from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferTransferOperationsListRequest(_messages.Message):
    """A StoragetransferTransferOperationsListRequest object.

  Fields:
    filter: Required. A list of query parameters specified as JSON text in the
      form of: `{"projectId":"my_project_id",
      "jobNames":["jobid1","jobid2",...], "jobNamePattern":
      "job_name_pattern", "operationNames":["opid1","opid2",...],
      "operationNamePattern": "operation_name_pattern", "minCreationTime":
      "min_creation_time", "maxCreationTime": "max_creation_time",
      "transferStatuses":["status1","status2",...]}` Since `jobNames`,
      `operationNames`, and `transferStatuses` support multiple values, they
      must be specified with array notation. `projectId` is the only argument
      that is required. If specified, `jobNamePattern` and
      `operationNamePattern` must match the full job or operation name
      respectively. '*' is a wildcard matching 0 or more characters.
      `minCreationTime` and `maxCreationTime` should be timestamps encoded as
      a string in the [RFC 3339](https://www.ietf.org/rfc/rfc3339.txt) format.
      The valid values for `transferStatuses` are case-insensitive:
      IN_PROGRESS, PAUSED, SUCCESS, FAILED, and ABORTED.
    name: Required. The name of the type being listed; must be
      `transferOperations`.
    pageSize: The list page size. The max allowed value is 256.
    pageToken: The list page token.
  """
    filter = _messages.StringField(1, required=True)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)