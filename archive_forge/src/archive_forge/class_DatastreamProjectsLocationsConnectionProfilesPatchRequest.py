from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsConnectionProfilesPatchRequest(_messages.Message):
    """A DatastreamProjectsLocationsConnectionProfilesPatchRequest object.

  Fields:
    connectionProfile: A ConnectionProfile resource to be passed as the
      request body.
    force: Optional. Update the connection profile without validating it.
    name: Output only. The resource's name.
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. The server will
      guarantee that for at least 60 minutes since the first request. For
      example, consider a situation where you make an initial request and the
      request times out. If you make the request again with the same request
      ID, the server can check if original operation with the same request ID
      was received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the ConnectionProfile resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all fields will be overwritten.
    validateOnly: Optional. Only validate the connection profile, but don't
      update any resources. The default is false.
  """
    connectionProfile = _messages.MessageField('ConnectionProfile', 1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    updateMask = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)