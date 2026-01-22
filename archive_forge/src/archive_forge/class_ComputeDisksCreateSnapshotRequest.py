from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeDisksCreateSnapshotRequest(_messages.Message):
    """A ComputeDisksCreateSnapshotRequest object.

  Fields:
    disk: Name of the persistent disk to snapshot.
    guestFlush: [Input Only] Whether to attempt an application consistent
      snapshot by informing the OS to prepare for the snapshot process.
    project: Project ID for this request.
    requestId: An optional request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. For example,
      consider a situation where you make an initial request and the request
      times out. If you make the request again with the same request ID, the
      server can check if original operation with the same request ID was
      received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      ( 00000000-0000-0000-0000-000000000000).
    snapshot: A Snapshot resource to be passed as the request body.
    zone: The name of the zone for this request.
  """
    disk = _messages.StringField(1, required=True)
    guestFlush = _messages.BooleanField(2)
    project = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    snapshot = _messages.MessageField('Snapshot', 5)
    zone = _messages.StringField(6, required=True)