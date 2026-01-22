from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesProjectsGlobalNetworksGetRequest(_messages.Message):
    """A ServicenetworkingServicesProjectsGlobalNetworksGetRequest object.

  Fields:
    includeUsedIpRanges: Optional. When true, include the used IP ranges as
      part of the GetConsumerConfig output. This includes routes created
      inside the service networking network, consumer network, peers of the
      consumer network, and reserved ranges inside the service networking
      network. By default, this is false
    name: Required. Name of the consumer config to retrieve in the format:
      `services/{service}/projects/{project}/global/networks/{network}`.
      {service} is the peering service that is managing connectivity for the
      service producer's organization. For Google services that support this
      functionality, this value is `servicenetworking.googleapis.com`.
      {project} is a project number e.g. `12345` that contains the service
      consumer's VPC network. {network} is the name of the service consumer's
      VPC network.
  """
    includeUsedIpRanges = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)