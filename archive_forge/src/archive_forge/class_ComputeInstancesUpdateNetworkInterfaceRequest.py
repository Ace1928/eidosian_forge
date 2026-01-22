from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstancesUpdateNetworkInterfaceRequest(_messages.Message):
    """A ComputeInstancesUpdateNetworkInterfaceRequest object.

  Fields:
    instance: The instance name for this request.
    networkInterface: The name of the network interface to update.
    networkInterfaceResource: A NetworkInterface resource to be passed as the
      request body.
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
    zone: The name of the zone for this request.
  """
    instance = _messages.StringField(1, required=True)
    networkInterface = _messages.StringField(2, required=True)
    networkInterfaceResource = _messages.MessageField('NetworkInterface', 3)
    project = _messages.StringField(4, required=True)
    requestId = _messages.StringField(5)
    zone = _messages.StringField(6, required=True)