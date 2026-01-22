from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import frozendict
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet.clusterupgrade import flags as clusterupgrade_flags
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
def GetFleetClusterUpgradeInfo(self, fleet, feature, args):
    """Gets Cluster Upgrade Feature information for the provided Fleet."""
    if args.IsKnownAndSpecified('show_linked_cluster_upgrade') and args.show_linked_cluster_upgrade:
        return self.GetLinkedClusterUpgrades(fleet, feature)
    return Describe.GetClusterUpgradeInfo(fleet, feature)