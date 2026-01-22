from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesProjectsGlobalNetworksDnsZonesListRequest(_messages.Message):
    """A ServicenetworkingServicesProjectsGlobalNetworksDnsZonesListRequest
  object.

  Fields:
    parent: Required. Parent resource identifying the connection which owns
      this collection of DNS zones in the format
      services/{service}/projects/{project}/global/networks/{network} Service:
      The service that is managing connectivity for the service producer's
      organization. For Google services that support this functionality, this
      value is `servicenetworking.googleapis.com`. Projects: the consumer
      project containing the consumer network. Network: The consumer network
      accessible from the tenant project.
  """
    parent = _messages.StringField(1, required=True)