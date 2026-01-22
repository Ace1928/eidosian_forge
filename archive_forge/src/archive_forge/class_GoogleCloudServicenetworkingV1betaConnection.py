from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudServicenetworkingV1betaConnection(_messages.Message):
    """Represents a private connection resource. A private connection is
  implemented as a VPC Network Peering connection between a service producer's
  VPC network and a service consumer's VPC network.

  Fields:
    network: The name of service consumer's VPC network that's connected with
      service producer network, in the following format:
      `projects/{project}/global/networks/{network}`. `{project}` is a project
      number, such as in `12345` that includes the VPC service consumer's VPC
      network. `{network}` is the name of the service consumer's VPC network.
    peering: Output only. The name of the VPC Network Peering connection that
      was created by the service producer.
    reservedPeeringRanges: The name of one or more allocated IP address ranges
      for this service producer of type `PEERING`. Note that invoking this
      method with a different range when connection is already established
      will not modify already provisioned service producer subnetworks.
    service: Output only. The name of the peering service that's associated
      with this connection, in the following format: `services/{service
      name}`.
  """
    network = _messages.StringField(1)
    peering = _messages.StringField(2)
    reservedPeeringRanges = _messages.StringField(3, repeated=True)
    service = _messages.StringField(4)