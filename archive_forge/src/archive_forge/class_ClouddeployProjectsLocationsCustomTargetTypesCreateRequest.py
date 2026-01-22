from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsCustomTargetTypesCreateRequest(_messages.Message):
    """A ClouddeployProjectsLocationsCustomTargetTypesCreateRequest object.

  Fields:
    customTargetType: A CustomTargetType resource to be passed as the request
      body.
    customTargetTypeId: Required. ID of the `CustomTargetType`.
    parent: Required. The parent collection in which the `CustomTargetType`
      should be created. Format should be
      `projects/{project_id}/locations/{location_name}`.
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server knows to
      ignore the request if it has already been completed. The server
      guarantees that for at least 60 minutes after the first request. For
      example, consider a situation where you make an initial request and the
      request times out. If you make the request again with the same request
      ID, the server can check if original operation with the same request ID
      was received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    validateOnly: Optional. If set to true, the request is validated and the
      user is provided with an expected result, but no actual change is made.
  """
    customTargetType = _messages.MessageField('CustomTargetType', 1)
    customTargetTypeId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)