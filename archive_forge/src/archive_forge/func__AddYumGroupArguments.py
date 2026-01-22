from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _AddYumGroupArguments(parser):
    """Adds Yum setting flags."""
    yum_group = parser.add_mutually_exclusive_group(help='Settings for machines running Yum:')
    non_exclusive_group = yum_group.add_group(help='Yum patch options')
    non_exclusive_group.add_argument('--yum-security', action='store_true', help='      If specified, machines running Yum append the `--security` flag to the\n      patch command.')
    non_exclusive_group.add_argument('--yum-minimal', action='store_true', help='      If specified, machines running Yum use the command `yum update-minimal`;\n      otherwise the patch uses `yum-update`.')
    non_exclusive_group.add_argument('--yum-excludes', metavar='YUM_EXCLUDES', type=arg_parsers.ArgList(), help='      Optional list of packages to exclude from updating. If this argument is\n      specified, machines running Yum exclude the given list of packages using\n      the Yum `--exclude` flag.')
    yum_group.add_argument('--yum-exclusive-packages', metavar='YUM_EXCLUSIVE_PACKAGES', type=arg_parsers.ArgList(), help='      An exclusive list of packages to be updated. These are the only packages\n      that will be updated. If these packages are not installed, they will be\n      ignored.')