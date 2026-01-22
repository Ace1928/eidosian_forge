from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
def GetAsyncSecondaryDiskUri(self, args, compute_holder):
    secondary_disk_ref = None
    if args.secondary_disk:
        secondary_disk_project = getattr(args, 'secondary_disk_project', None)
        secondary_disk_ref = self.secondary_disk_arg.ResolveAsResource(args, compute_holder.resources, source_project=secondary_disk_project)
        if secondary_disk_ref:
            return secondary_disk_ref.SelfLink()
    return None