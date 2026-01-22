from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
def AddPropagatedConnectionLimit(parser):
    parser.add_argument('--propagated-connection-limit', type=int, help='    The number of consumer spokes that connected Private Service Connect\n    endpoints can be propagated to through Network Connectivity Center. This\n    limit lets the service producer limit how many propagated Private Service\n    Connect connections can be established to this service attachment from a\n    single consumer.\n\n    If the connection preference of the service attachment is ACCEPT_MANUAL, the\n    limit applies to each project or network that is listed in the consumer\n    accept list. If the connection preference of the service attachment is\n    ACCEPT_AUTOMATIC, the limit applies to each project that contains a\n    connected endpoint.\n\n    If unspecified, the default propagated connection limit is 250.\n    ')