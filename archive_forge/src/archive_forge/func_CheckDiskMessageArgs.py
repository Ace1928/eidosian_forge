from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def CheckDiskMessageArgs(self, args, skip_defaults):
    """Creates API messages with disks attached to VM instance."""
    flags_to_check = ['create_disk', 'local_ssd', 'boot_disk_type', 'boot_disk_device_name', 'boot_disk_auto_delete', 'boot_disk_provisioned_iops']
    if hasattr(args, 'local_nvdimm'):
        flags_to_check.append('local_nvdimm')
    if skip_defaults and (not args.IsSpecified('disk')) and (not instance_utils.IsAnySpecified(args, *flags_to_check)):
        return False
    return True