from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def GetInstanceGroupManagerArg(zones_flag=False, region_flag=True):
    """Returns ResourceArgument for working with instance group managers."""
    if zones_flag:
        extra_region_info_about_zones_flag = '\n\nIf you specify `--zones` flag this flag must be unspecified or specify the region to which the zones you listed belong.'
        region_explanation = flags.REGION_PROPERTY_EXPLANATION_NO_DEFAULT + extra_region_info_about_zones_flag
    else:
        region_explanation = flags.REGION_PROPERTY_EXPLANATION_NO_DEFAULT
    if region_flag:
        regional_collection = 'compute.regionInstanceGroupManagers'
    else:
        regional_collection = None
    return flags.ResourceArgument(resource_name='managed instance group', completer=InstanceGroupManagersCompleter, zonal_collection='compute.instanceGroupManagers', regional_collection=regional_collection, zone_explanation=flags.ZONE_PROPERTY_EXPLANATION_NO_DEFAULT, region_explanation=region_explanation)