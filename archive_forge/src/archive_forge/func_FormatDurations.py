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
@staticmethod
def FormatDurations(cluster_upgrade_spec):
    """Formats display strings for all cluster upgrade duration fields."""
    if cluster_upgrade_spec.postConditions is not None:
        default_soaking = cluster_upgrade_spec.postConditions.soaking
        if default_soaking is not None:
            cluster_upgrade_spec.postConditions.soaking = Describe.DisplayDuration(default_soaking)
    for override in cluster_upgrade_spec.gkeUpgradeOverrides:
        if override.postConditions is not None:
            override_soaking = override.postConditions.soaking
            if override_soaking is not None:
                override.postConditions.soaking = Describe.DisplayDuration(override_soaking)
    return cluster_upgrade_spec