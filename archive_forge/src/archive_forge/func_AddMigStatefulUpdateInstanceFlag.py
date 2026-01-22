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
def AddMigStatefulUpdateInstanceFlag(parser):
    """Add flags for applying updates on PIC change."""
    parser.add_argument('--update-instance', default=True, action='store_true', help='\n          Apply the configuration changes immediately to the instance. If you\n          disable this flag, the managed instance group will apply the\n          configuration update when you next recreate or update the instance.\n\n          Example: say you have an instance with a disk attached to it and you\n          created a stateful configuration for the disk. If you decide to\n          delete the stateful configuration for the disk and you provide this\n          flag, the group immediately refreshes the instance and removes the\n          stateful configuration for the disk. Similarly if you have attached\n          a new disk or changed its definition, with this flag the group\n          immediately refreshes the instance with the new configuration.')
    parser.add_argument('--instance-update-minimal-action', choices=mig_flags.InstanceActionChoicesWithNone(), default='none', help='\n          Perform at least this action on the instance while updating, if\n          `--update-instance` is set to `true`.')