from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddDiskArgs(parser, enable_regional_disks=False, enable_kms=False, container_mount_enabled=False):
    """Adds arguments related to disks for instances and instance-templates."""
    disk_device_name_help = GetDiskDeviceNameHelp(container_mount_enabled=container_mount_enabled)
    AddBootDiskArgs(parser, enable_kms)
    disk_arg_spec = {'name': str, 'mode': str, 'boot': arg_parsers.ArgBoolean(), 'device-name': str, 'auto-delete': arg_parsers.ArgBoolean()}
    if enable_regional_disks:
        disk_arg_spec['scope'] = str
        disk_arg_spec['force-attach'] = arg_parsers.ArgBoolean()
    disk_help = "\n      Attaches an existing persistent disk to the instances.\n\n      *name*::: The disk to attach to the instances. If you create more than\n      one instance, you can only attach a disk in read-only mode. By default,\n      you attach a zonal persistent disk located in the same zone of the\n      instance. If you want to attach a regional persistent disk, you must\n      specify the disk using its URI; for example,\n      ``projects/myproject/regions/us-central1/disks/my-regional-disk''.\n\n      *mode*::: The mode of the disk. Supported options are ``ro'' for read-only\n      mode and ``rw'' for read-write mode. If omitted, ``rw'' is used as\n      a default value. If you use ``rw'' when creating more than one instance,\n      you encounter errors.\n\n      *boot*::: If set to ``yes'', you attach a boot disk. The\n      virtual machine then uses the first partition of the disk for\n      the root file systems. The default value for this is ``no''.\n\n      *device-name*::: {}\n\n      *auto-delete*::: If set to ``yes'', the persistent disk is\n      automatically deleted when the instance is deleted. However,\n      if you detach the disk from the instance, deleting the instance\n      doesn't delete the disk. The default value for this is ``yes''.\n      ".format(disk_device_name_help)
    if enable_regional_disks:
        disk_help += "\n      *scope*::: Can be `zonal` or `regional`. If ``zonal'', the disk is\n      interpreted as a zonal disk in the same zone as the instance (default).\n      If ``regional'', the disk is interpreted as a regional disk in the same\n      region as the instance. The default value for this is ``zonal''.\n      *force-attach*::: If ``yes'',  this persistent disk will force-attached to\n      the instance even it is already attached to another instance. The default\n      value is 'no'.\n      "
    parser.add_argument('--disk', type=arg_parsers.ArgDict(spec=disk_arg_spec), action='append', help=disk_help)