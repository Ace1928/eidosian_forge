from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
def AddReconcileConnectionsForUpdate(parser):
    parser.add_argument('--reconcile-connections', action=arg_parsers.StoreTrueFalseAction, help='      Determines whether to apply changes to consumer accept or reject lists\n      to existing connections or only to new connections.\n\n      If false, existing endpoints with a connection status of ACCEPTED or\n      REJECTED are not updated.\n\n      If true, existing endpoints with a connection status of ACCEPTED or\n      REJECTED are updated based on the connection policy update. For example,\n      if a project or network is removed from the --consumer-accept-list and\n      added to --consumer-reject-list, all the endpoints in that project or\n      network with the ACCEPTED state are set to REJECTED.\n      ')