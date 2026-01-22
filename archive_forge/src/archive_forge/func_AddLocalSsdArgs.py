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
def AddLocalSsdArgs(parser):
    """Adds local SSD argument for instances and instance-templates."""
    parser.add_argument('--local-ssd', type=arg_parsers.ArgDict(spec={'device-name': str, 'interface': lambda x: x.upper()}), action='append', help="      Attaches a local SSD to the instances.\n\n      *device-name*::: Optional. A name that indicates the disk name\n      the guest operating system will see. Can only be specified if\n      `interface` is `SCSI`. If omitted, a device name\n      of the form ``local-ssd-N'' will be used.\n\n      *interface*::: Optional. The kind of disk interface exposed to the VM\n      for this SSD.  Valid values are ``SCSI'' and ``NVME''.  SCSI is\n      the default and is supported by more guest operating systems.  NVME\n      might provide higher performance.\n      ")