from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes as compute_base
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetComputeResources(release_track, cluster_name, dataproc_region):
    """Returns a resources object with resolved GCE zone and region."""
    holder = compute_base.ComputeApiHolder(release_track)
    region_prop = properties.VALUES.compute.region
    zone_prop = properties.VALUES.compute.zone
    resources = holder.resources
    zone = properties.VALUES.compute.zone.Get()
    if not zone and dataproc_region == 'global':
        _, zone = scope_prompter.PromptForScope(resource_name='cluster', underspecified_names=[cluster_name], scopes=[compute_scope.ScopeEnum.ZONE], default_scope=None, scope_lister=flags.GetDefaultScopeLister(holder.client))
        if not zone:
            zone = properties.VALUES.compute.zone.GetOrFail()
    if zone:
        zone_ref = resources.Parse(zone, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.zones')
        zone_name = zone_ref.Name()
        zone_prop.Set(zone_name)
        region_name = compute_utils.ZoneNameToRegionName(zone_name)
        region_prop.Set(region_name)
    else:
        zone_prop.Set('')
        region_prop.Set(dataproc_region)
    return resources