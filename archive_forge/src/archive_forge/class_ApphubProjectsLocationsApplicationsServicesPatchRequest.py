from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsApplicationsServicesPatchRequest(_messages.Message):
    """A ApphubProjectsLocationsApplicationsServicesPatchRequest object.

  Fields:
    name: Identifier. The resource name of a Service. Format: "projects/{host-
      project-id}/locations/{location}/applications/{application-
      id}/services/{service-id}"
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
    service: A Service resource to be passed as the request body.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the Service resource by the update. The fields specified
      in the update_mask are relative to the resource, not the full request.
      The API changes the values of the fields as specified in the
      update_mask. The API ignores the values of all fields not covered by the
      update_mask. You can also unset a field by not specifying it in the
      updated message, but adding the field to the mask. This clears whatever
      value the field previously had.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    service = _messages.MessageField('Service', 3)
    updateMask = _messages.StringField(4)