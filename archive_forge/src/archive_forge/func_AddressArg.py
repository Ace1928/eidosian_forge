from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def AddressArg():
    return compute_flags.ResourceArgument(name='--address', required=False, resource_name='address', completer=addresses_flags.AddressesCompleter, regional_collection='compute.addresses', global_collection='compute.globalAddresses', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION, short_help='IP address that the forwarding rule will serve.', detailed_help=AddressArgHelp())