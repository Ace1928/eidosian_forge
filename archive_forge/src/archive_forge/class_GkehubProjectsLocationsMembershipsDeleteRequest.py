from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsDeleteRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsDeleteRequest object.

  Fields:
    force: Optional. If set to true, any subresource from this Membership will
      also be deleted. Otherwise, the request will only work if the Membership
      has no subresource.
    name: Required. The Membership resource name in the format
      `projects/*/locations/*/memberships/*`.
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. The server will
      guarantee that for at least 60 minutes after the first request. For
      example, consider a situation where you make an initial request and the
      request times out. If you make the request again with the same request
      ID, the server can check if original operation with the same request ID
      was received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)