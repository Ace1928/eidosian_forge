from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesConnectionsDeleteConnectionRequest(_messages.Message):
    """A ServicenetworkingServicesConnectionsDeleteConnectionRequest object.

  Fields:
    deleteConnectionRequest: A DeleteConnectionRequest resource to be passed
      as the request body.
    name: Required. The private service connection that connects to a service
      producer organization. The name includes both the private service name
      and the VPC network peering name in the format of
      `services/{peering_service_name}/connections/{vpc_peering_name}`. For
      Google services that support this functionality, this is `services/servi
      cenetworking.googleapis.com/connections/servicenetworking-googleapis-
      com`.
  """
    deleteConnectionRequest = _messages.MessageField('DeleteConnectionRequest', 1)
    name = _messages.StringField(2, required=True)