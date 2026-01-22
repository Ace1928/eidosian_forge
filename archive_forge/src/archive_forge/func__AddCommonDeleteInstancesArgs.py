from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from the managed instance group, use the abandon-instances command instead.
def _AddCommonDeleteInstancesArgs(parser):
    """Add parser configuration common for all release tracks."""
    parser.display_info.AddFormat(mig_flags.GetCommonPerInstanceCommandOutputFormat())
    parser.add_argument('--instances', type=arg_parsers.ArgList(min_length=1), metavar='INSTANCE', required=True, help='Names of instances to delete.')
    instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)
    mig_flags.AddGracefulValidationArg(parser)