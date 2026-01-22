from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddressArgument(required=True, plural=True):
    return compute_flags.ResourceArgument(resource_name='address', completer=AddressesCompleter, plural=plural, custom_plural='addresses', required=required, regional_collection='compute.addresses', global_collection='compute.globalAddresses')