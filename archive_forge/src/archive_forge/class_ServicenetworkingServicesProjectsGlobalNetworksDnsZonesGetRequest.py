from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesProjectsGlobalNetworksDnsZonesGetRequest(_messages.Message):
    """A ServicenetworkingServicesProjectsGlobalNetworksDnsZonesGetRequest
  object.

  Fields:
    name: Required. The network that the consumer is using to connect with
      services. Must be in the form of services/{service}/projects/{project}/g
      lobal/networks/{network}/zones/{zoneName} Where {service} is the peering
      service that is managing connectivity for the service producer's
      organization. For Google services that support this {project} is the
      project number, as in '12345' {network} is the network name. {zoneName}
      is the DNS zone name
  """
    name = _messages.StringField(1, required=True)