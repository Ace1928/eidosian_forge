from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.networks.subnets import flags
import six
def CreateSubnetworkPatchRequest(client, subnet_ref, subnetwork_resource):
    patch_request = client.messages.ComputeSubnetworksPatchRequest(project=subnet_ref.project, subnetwork=subnet_ref.subnetwork, region=subnet_ref.region, subnetworkResource=subnetwork_resource)
    return (client.apitools_client.subnetworks, 'Patch', patch_request)