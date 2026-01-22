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
def _CreateGroupReference(self, args, client, resources):
    if args.zones:
        zone_ref = resources.Parse(args.zones[0], collection='compute.zones', params={'project': properties.VALUES.core.project.GetOrFail})
        region = utils.ZoneNameToRegionName(zone_ref.Name())
        return resources.Parse(args.name, params={'region': region, 'project': properties.VALUES.core.project.GetOrFail}, collection='compute.regionInstanceGroupManagers')
    group_ref = instance_groups_flags.GetInstanceGroupManagerArg().ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.ZONE, scope_lister=flags.GetDefaultScopeLister(client))
    if _IsZonalGroup(group_ref):
        zonal_resource_fetcher = zone_utils.ZoneResourceFetcher(client)
        zonal_resource_fetcher.WarnForZonalCreation([group_ref])
    return group_ref