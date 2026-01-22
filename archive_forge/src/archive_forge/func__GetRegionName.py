from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
def _GetRegionName(self, igm_ref):
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        return utils.ZoneNameToRegionName(igm_ref.zone)
    elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        return igm_ref.region
    else:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))