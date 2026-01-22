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
@staticmethod
def _HandleUpstreamFleets(args, cluster_upgrade_spec):
    """Updates the Cluster Upgrade Feature's upstreamFleets field."""
    if args.IsKnownAndSpecified('reset_upstream_fleet') and args.reset_upstream_fleet:
        cluster_upgrade_spec.upstreamFleets = []
    elif args.IsKnownAndSpecified('upstream_fleet') and args.upstream_fleet is not None:
        cluster_upgrade_spec.upstreamFleets = [args.upstream_fleet]