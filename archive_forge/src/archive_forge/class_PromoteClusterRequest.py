from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PromoteClusterRequest(_messages.Message):
    """Message for promoting a Cluster

  Fields:
    etag: Optional. The current etag of the Cluster. If an etag is provided
      and does not match the current etag of the Cluster, deletion will be
      blocked and an ABORTED error will be returned.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes after the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    validateOnly: Optional. If set, performs request validation (e.g.
      permission checks and any other type of validation), but do not actually
      execute the delete.
  """
    etag = _messages.StringField(1)
    requestId = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)