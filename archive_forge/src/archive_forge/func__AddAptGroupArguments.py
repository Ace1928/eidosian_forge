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
def _AddAptGroupArguments(parser):
    """Adds Apt setting flags."""
    apt_group = parser.add_group(help='Settings for machines running Apt:')
    apt_group.add_argument('--apt-dist', action='store_true', help='      If specified, machines running Apt use the `apt-get dist-upgrade` command;\n      otherwise the `apt-get upgrade` command is used.')
    mutually_exclusive_group = apt_group.add_mutually_exclusive_group()
    mutually_exclusive_group.add_argument('--apt-excludes', metavar='APT_EXCLUDES', type=arg_parsers.ArgList(), help='List of packages to exclude from update.')
    mutually_exclusive_group.add_argument('--apt-exclusive-packages', metavar='APT_EXCLUSIVE_PACKAGES', type=arg_parsers.ArgList(), help='      An exclusive list of packages to be updated. These are the only packages\n      that will be updated. If these packages are not installed, they will be\n      ignored.')