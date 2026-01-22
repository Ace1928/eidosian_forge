from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def MakeDiskArg(plural):
    return compute_flags.ResourceArgument(resource_name='disk', completer=compute_completers.DisksCompleter, plural=plural, name='DISK_NAME', zonal_collection='compute.disks', regional_collection='compute.regionDisks', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION, region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)