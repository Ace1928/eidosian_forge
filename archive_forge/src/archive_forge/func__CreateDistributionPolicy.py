from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import stateful_policy_utils as policy_utils
from googlecloudsdk.api_lib.compute.managed_instance_groups_utils import ValueOrNone
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def _CreateDistributionPolicy(self, args, resources, messages):
    distribution_policy = messages.DistributionPolicy()
    if args.zones:
        policy_zones = []
        for zone in args.zones:
            zone_ref = resources.Parse(zone, collection='compute.zones', params={'project': properties.VALUES.core.project.GetOrFail})
            policy_zones.append(messages.DistributionPolicyZoneConfiguration(zone=zone_ref.SelfLink()))
        distribution_policy.zones = policy_zones
    if args.target_distribution_shape:
        distribution_policy.targetShape = arg_utils.ChoiceToEnum(args.target_distribution_shape, messages.DistributionPolicy.TargetShapeValueValuesEnum)
    return ValueOrNone(distribution_policy)