from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def AddDiskArgsForBulk(parser):
    """Adds arguments related to disks for bulk insert."""
    disk_device_name_help = instances_flags.GetDiskDeviceNameHelp(container_mount_enabled=False)
    instances_flags.AddBootDiskArgs(parser, enable_kms=True)
    disk_arg_spec = {'name': str, 'boot': arg_parsers.ArgBoolean(), 'device-name': str, 'scope': str}
    disk_help = "\n      Attaches persistent disks to the instances. The disks\n      specified must already exist.\n\n      *name*::: The disk to attach to the instances.\n\n      *boot*::: If ``yes'', indicates that this is a boot disk. The\n      virtual machines will use the first partition of the disk for\n      their root file systems. The default value for this is ``no''.\n\n      *device-name*::: {}\n\n      *scope*::: Can be `zonal` or `regional`. If ``zonal'', the disk is\n      interpreted as a zonal disk in the same zone as the instance (default).\n      If ``regional'', the disk is interpreted as a regional disk in the same\n      region as the instance. The default value for this is ``zonal''.\n      ".format(disk_device_name_help)
    parser.add_argument('--disk', type=arg_parsers.ArgDict(spec=disk_arg_spec), action='append', help=disk_help)