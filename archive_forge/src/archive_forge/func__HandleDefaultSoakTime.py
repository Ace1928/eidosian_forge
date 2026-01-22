from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet.clusterupgrade import flags as clusterupgrade_flags
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def _HandleDefaultSoakTime(self, args, cluster_upgrade_spec):
    """Updates the Cluster Upgrade Feature's postConditions.soaking field."""
    if not args.IsKnownAndSpecified('default_upgrade_soaking') or args.default_upgrade_soaking is None:
        return
    default_soaking = times.FormatDurationForJson(iso_duration.Duration(seconds=args.default_upgrade_soaking))
    post_conditions = cluster_upgrade_spec.postConditions or self.messages.ClusterUpgradePostConditions()
    post_conditions.soaking = default_soaking
    cluster_upgrade_spec.postConditions = post_conditions