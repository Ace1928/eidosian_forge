from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.networks.subnets import flags
import six
def _CreateSecondaryRange(client, name, ip_cidr_range, reserved_internal_range):
    """Creates a subnetwork secondary range."""
    if reserved_internal_range and ip_cidr_range:
        return client.messages.SubnetworkSecondaryRange(rangeName=name, reservedInternalRange=reserved_internal_range, ipCidrRange=ip_cidr_range)
    elif reserved_internal_range:
        return client.messages.SubnetworkSecondaryRange(rangeName=name, reservedInternalRange=reserved_internal_range)
    else:
        return client.messages.SubnetworkSecondaryRange(rangeName=name, ipCidrRange=ip_cidr_range)