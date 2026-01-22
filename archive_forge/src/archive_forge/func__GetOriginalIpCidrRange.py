from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.networks.subnets import flags
from googlecloudsdk.core.console import console_io
import ipaddress
import six
def _GetOriginalIpCidrRange(self, client, subnetwork_ref):
    subnetwork = self._GetSubnetwork(client, subnetwork_ref)
    if not subnetwork:
        raise compute_exceptions.ArgumentError('Subnet [{subnet}] was not found in region {region}.'.format(subnet=subnetwork_ref.Name(), region=subnetwork_ref.region))
    return subnetwork.ipCidrRange