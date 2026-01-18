from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1beta import servicenetworking_v1beta_messages as messages
Service producers use this method to provision a new subnet in.
peered service shared VPC network.
It will validate previously provided reserved ranges, find
non-conflicting sub-range of requested size (expressed in
number of leading bits of ipv4 network mask, as in CIDR range
notation). It will then create a subnetwork in the request
region. Operation<response: Subnetwork>

      Args:
        request: (ServicenetworkingServicesAddSubnetworkRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      