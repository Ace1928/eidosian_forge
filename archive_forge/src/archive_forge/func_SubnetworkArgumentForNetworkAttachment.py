from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.util.apis import arg_utils
def SubnetworkArgumentForNetworkAttachment(required=True):
    return compute_flags.ResourceArgument(resource_name='subnetwork', name='--subnets', completer=SubnetworksCompleter, plural=True, required=required, regional_collection='compute.subnetworks', short_help='The subnetworks provided by the consumer for the producers')