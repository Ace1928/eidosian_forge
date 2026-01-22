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
def _DownTheStream(cluster_upgrade):
    """Recursively gets information for the downstream Fleets."""
    downstream_state = cluster_upgrade.get('state', None)
    downstream_fleets = downstream_state.downstreamFleets if downstream_state else None
    if not downstream_fleets:
        return [cluster_upgrade]
    downstream_fleet = downstream_fleets[0]
    if downstream_fleet in visited:
        return [cluster_upgrade]
    visited.add(downstream_fleet)
    downstream_scope_project = Describe.GetProjectIDFromFleet(downstream_fleet)
    downstream_feature = feature if downstream_scope_project == current_project else self.GetFeature(project=downstream_scope_project)
    downstream_cluster_upgrade = Describe.GetClusterUpgradeInfo(downstream_fleet, downstream_feature)
    return [cluster_upgrade] + _DownTheStream(downstream_cluster_upgrade)