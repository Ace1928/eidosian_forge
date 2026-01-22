from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetNetworkEndpointGroupArg(support_global_neg=False, support_region_neg=False):
    return compute_flags.ResourceArgument(name='--network-endpoint-group', resource_name='network endpoint group', zonal_collection='compute.networkEndpointGroups', global_collection='compute.globalNetworkEndpointGroups' if support_global_neg else None, regional_collection='compute.regionNetworkEndpointGroups' if support_region_neg else None, zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION, region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION if support_region_neg else None)