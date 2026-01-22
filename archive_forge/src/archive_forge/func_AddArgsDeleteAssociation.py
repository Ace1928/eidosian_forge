from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsDeleteAssociation(parser):
    """Adds the arguments of association deletion."""
    parser.add_argument('--firewall-policy', required=True, help='Short name or ID of the firewall policy ID of the association.')
    parser.add_argument('--organization', help='ID of the organization in which the firewall policy is to be detached. Must be set if FIREWALL_POLICY is short name.')