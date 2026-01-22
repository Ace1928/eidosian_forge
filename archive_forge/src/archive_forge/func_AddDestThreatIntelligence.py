from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddDestThreatIntelligence(parser):
    """Adds destination threat intelligence list names to this rule."""
    parser.add_argument('--dest-threat-intelligence', type=arg_parsers.ArgList(), metavar='DEST_THREAT_INTELLIGENCE_LISTS', required=False, help='Destination Threat Intelligence lists to match for this rule. Can only be specified if DIRECTION is `egress`. The available lists can be found here: https://cloud.google.com/vpc/docs/firewall-policies-rule-details#threat-intelligence-fw-policy.')