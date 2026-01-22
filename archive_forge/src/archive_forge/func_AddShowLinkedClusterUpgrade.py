from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def AddShowLinkedClusterUpgrade(self):
    """Adds the --show-linked-cluster-upgrade flag."""
    self.parser.add_argument('--show-linked-cluster-upgrade', action='store_true', default=None, help="        Shows the cluster upgrade feature information for the current fleet as\n        well as information for all other fleets linked in the same rollout\n        sequence (provided that the caller has permission to view the upstream\n        and downstream fleets). This displays cluster upgrade information for\n        fleets in the current fleet's rollout sequence in order of furthest\n        upstream to downstream.\n\n        To view the cluster upgrade feature information for the rollout\n        sequence containing the current fleet, run:\n\n          $ {command} --show-linked-cluster-upgrade\n        ")