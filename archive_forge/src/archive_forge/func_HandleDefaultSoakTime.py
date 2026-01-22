from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
def HandleDefaultSoakTime(self, cluster_upgrade_spec):
    """Updates the Cluster Upgrade Feature's postConditions.soaking field."""
    if not self.args.IsKnownAndSpecified('default_upgrade_soaking') or self.args.default_upgrade_soaking is None:
        return
    default_soaking = times.FormatDurationForJson(self.args.default_upgrade_soaking)
    post_conditions = cluster_upgrade_spec.postConditions or self.messages.ClusterUpgradePostConditions()
    post_conditions.soaking = default_soaking
    cluster_upgrade_spec.postConditions = post_conditions