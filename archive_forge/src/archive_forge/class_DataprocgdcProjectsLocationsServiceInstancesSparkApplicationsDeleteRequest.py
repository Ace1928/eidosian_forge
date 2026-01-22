from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsDeleteRequest(_messages.Message):
    """A
  DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsDeleteRequest
  object.

  Fields:
    allowMissing: Optional. If set to true, and the application is not found,
      the request will succeed but no action will be taken on the server
    etag: Optional. The etag of the application. If this is provided, it must
      match the server etag.
    name: Required. The name of the application to delete.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)