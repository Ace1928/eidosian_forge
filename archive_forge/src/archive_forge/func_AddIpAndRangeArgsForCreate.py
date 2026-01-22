from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddIpAndRangeArgsForCreate(parser):
    """Adds arguments to specify source NAT IP Addresses when creating a rule."""
    ACTIVE_IPS_ARG_OPTIONAL.AddArgument(parser, cust_metavar='IP_ADDRESS')
    ACTIVE_RANGES_ARG.AddArgument(parser, cust_metavar='SUBNETWORK')