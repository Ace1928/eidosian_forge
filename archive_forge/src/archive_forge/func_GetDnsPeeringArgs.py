from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetDnsPeeringArgs():
    """Return arg group for DNS Peering flags."""
    peering_group = base.ArgumentGroup(required=False)
    target_network_help_text = 'Network ID of the Google Compute Engine private network to forward queries to.'
    target_project_help_text = 'Project ID of the Google Compute Engine private network to forward queries to.'
    peering_group.AddArgument(base.Argument('--target-network', required=True, help=target_network_help_text))
    peering_group.AddArgument(base.Argument('--target-project', required=True, help=target_project_help_text))
    return peering_group