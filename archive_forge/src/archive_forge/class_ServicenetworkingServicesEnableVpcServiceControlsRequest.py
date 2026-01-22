from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesEnableVpcServiceControlsRequest(_messages.Message):
    """A ServicenetworkingServicesEnableVpcServiceControlsRequest object.

  Fields:
    enableVpcServiceControlsRequest: A EnableVpcServiceControlsRequest
      resource to be passed as the request body.
    parent: The service that is managing peering connectivity for a service
      producer's organization. For Google services that support this
      functionality, this value is
      `services/servicenetworking.googleapis.com`.
  """
    enableVpcServiceControlsRequest = _messages.MessageField('EnableVpcServiceControlsRequest', 1)
    parent = _messages.StringField(2, required=True)