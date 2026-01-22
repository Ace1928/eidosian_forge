from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core.util.iso_duration import Duration
from googlecloudsdk.core.util.times import FormatDuration
def MakeQueuedResourcesArg(plural=False):
    return compute_flags.ResourceArgument(resource_name='queued resource', zonal_collection='compute.zoneQueuedResources', plural=plural, zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION)