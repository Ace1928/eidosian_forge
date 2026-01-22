from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeNetworkEndpointGroupsArg():
    return compute_flags.ResourceArgument(resource_name='network endpoint group', zonal_collection='compute.networkEndpointGroups', global_collection='compute.globalNetworkEndpointGroups', regional_collection='compute.regionNetworkEndpointGroups', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION, region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)