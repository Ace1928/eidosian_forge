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
def _AddZypperGroupArguments(parser):
    """Adds Zypper setting flags."""
    zypper_group = parser.add_mutually_exclusive_group(help='Settings for machines running Zypper:')
    non_exclusive_group = zypper_group.add_group('Zypper patch options')
    non_exclusive_group.add_argument('--zypper-categories', metavar='ZYPPER_CATEGORIES', type=arg_parsers.ArgList(), help='      If specified, machines running Zypper install only patches with the\n      specified categories. Categories include security, recommended, and\n      feature.')
    non_exclusive_group.add_argument('--zypper-severities', metavar='ZYPPER_SEVERITIES', type=arg_parsers.ArgList(), help='      If specified, machines running Zypper install only patch with the\n      specified severities. Severities include critical, important, moderate,\n      and low.')
    non_exclusive_group.add_argument('--zypper-with-optional', action='store_true', help='      If specified, machines running Zypper add the `--with-optional` flag to\n      `zypper patch`.')
    non_exclusive_group.add_argument('--zypper-with-update', action='store_true', help='      If specified, machines running Zypper add the `--with-update` flag to\n      `zypper patch`.')
    non_exclusive_group.add_argument('--zypper-excludes', metavar='ZYPPER_EXCLUDES', type=arg_parsers.ArgList(), help='      List of Zypper patches to exclude from the patch job.\n      ')
    zypper_group.add_argument('--zypper-exclusive-patches', metavar='ZYPPER_EXCLUSIVE_PATCHES', type=arg_parsers.ArgList(), help="      An exclusive list of patches to be updated. These are the only patches\n      that will be installed using the 'zypper patch patch:<patch_name>'\n      command.")