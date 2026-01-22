from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsComplexTypesPatchRequest(_messages.Message):
    """A MediaassetProjectsLocationsComplexTypesPatchRequest object.

  Fields:
    complexType: A ComplexType resource to be passed as the request body.
    name: The resource name of the complex type, in the following form:
      `projects/{project}/locations/{location}/complexTypes/{type}`. Here
      {type} is a resource id. Detailed rules for a resource id are: 1. 1
      character minimum, 63 characters maximum 2. only contains letters,
      digits, underscore and hyphen 3. starts with a letter if length == 1,
      starts with a letter or underscore if length > 1
    requestId: An optional request ID to identify requests. Specify a unique
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
    updateMask: Required. Comma-separated list of fields is used to specify
      the fields to be overwritten in the ComplexType resource by the update.
      The fields specified in the update_mask are relative to the resource,
      not the full request. A field will be overwritten if it is in the mask.
      If the user does not provide a mask then all fields will be overwritten.
  """
    complexType = _messages.MessageField('ComplexType', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)