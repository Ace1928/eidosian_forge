from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateInstanceResizeDisk(args, messages):
    """Create and return ResizeDisk request."""
    instance = GetInstanceResource(args).RelativeName()
    request = None
    if args.IsSpecified('boot_disk_size'):
        request = messages.ResizeDiskRequest(bootDisk=messages.BootDisk(diskSizeGb=args.boot_disk_size))
    elif args.IsSpecified('data_disk_size'):
        request = messages.ResizeDiskRequest(dataDisk=messages.DataDisk(diskSizeGb=args.data_disk_size))
    return messages.NotebooksProjectsLocationsInstancesResizeDiskRequest(notebookInstance=instance, resizeDiskRequest=request)