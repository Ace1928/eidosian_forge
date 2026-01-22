from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def AddMigUpdateStatefulFlags(parser):
    """Add --stateful-disk and --remove-stateful-disks to the parser."""
    stateful_disks_help = textwrap.dedent(STATEFUL_DISKS_HELP_BASE + '\n      Use this argument multiple times to update more disks.\n\n      If a stateful disk with the given device name already exists in the\n      current instance configuration, its properties will be replaced by the\n      newly provided ones. Otherwise, a new stateful disk definition will be\n      added to the instance configuration.\n\n      *device-name*::: (Required) Device name of the disk to mark stateful.\n      ' + STATEFUL_DISK_AUTO_DELETE_ARG_HELP)
    stateful_disk_flag_name = '--stateful-disk'
    parser.add_argument(stateful_disk_flag_name, type=arg_parsers.ArgDict(spec={'device-name': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(stateful_disk_flag_name)}), action='append', help=stateful_disks_help)
    parser.add_argument('--remove-stateful-disks', metavar='DEVICE_NAME', type=arg_parsers.ArgList(min_length=1), help='Remove stateful configuration for the specified disks.')