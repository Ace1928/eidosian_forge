from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferapplianceProjectsLocationsOrdersPatchRequest(_messages.Message):
    """A TransferapplianceProjectsLocationsOrdersPatchRequest object.

  Fields:
    allowMissing: Optional. If set to true, updating a `Order` that does not
      exist will result in the creation of a new `Order`.
    name: name of resource.
    order: A Order resource to be passed as the request body.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and t he request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the Order resource by the update. The fields specified in
      the update_mask are relative to the resource, not the full request. A
      field will be overwritten if it is in the mask. If the user does not
      provide a mask then all populated fields will be overwritten.
    validateOnly: Optional. If set to true, the request is validated and the
      user is provided with an expected result, but no actual change is made.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    order = _messages.MessageField('Order', 3)
    requestId = _messages.StringField(4)
    updateMask = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)