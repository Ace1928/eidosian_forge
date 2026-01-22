from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core.console import console_io
from six.moves import zip
def AutoDeleteMustBeChanged(self, args, disk_resource):
    """Returns True if the autoDelete property of the disk must be changed."""
    if args.keep_disks == 'boot':
        return disk_resource.autoDelete and disk_resource.boot
    elif args.keep_disks == 'data':
        return disk_resource.autoDelete and (not disk_resource.boot)
    elif args.keep_disks == 'all':
        return disk_resource.autoDelete
    elif args.delete_disks == 'data':
        return not disk_resource.autoDelete and (not disk_resource.boot)
    elif args.delete_disks == 'all':
        return not disk_resource.autoDelete
    elif args.delete_disks == 'boot':
        return not disk_resource.autoDelete and disk_resource.boot
    return False