from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCommonNatArgs(parser, for_create=False):
    """Adds common arguments for creating and updating NATs."""
    _AddAutoNetworkTier(parser)
    _AddIpAllocationArgs(parser)
    _AddSubnetworkArgs(parser, for_create)
    _AddTimeoutsArgs(parser, for_create)
    _AddMinPortsPerVmArg(parser, for_create)
    _AddLoggingArgs(parser)
    _AddEndpointIndependentMappingArg(parser)
    if not for_create:
        _AddDrainNatIpsArgument(parser)
    _AddRulesArg(parser)
    _AddDynamicPortAllocationArgs(parser, for_create)