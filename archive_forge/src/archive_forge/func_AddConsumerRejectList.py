from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
def AddConsumerRejectList(parser):
    parser.add_argument('--consumer-reject-list', type=arg_parsers.ArgList(), metavar='REJECT_LIST', default=None, help='      Specifies a comma separated list of projects or networks that are not\n      allowed to connect to this service attachment. The project can be\n      specified using its project ID or project number and the network can be\n      specified using its URL. A given service attachment can manage connections\n      at either the project or network level. Therefore, both the reject and\n      accept lists for a given service attachment must contain either only\n      projects or only networks.')