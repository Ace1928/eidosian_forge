from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddFilterArgForUpdate(flag_config):
    """Adds trigger filter flag arg.

  Args:
    flag_config: argparse argument group. Trigger filter flag will be added to
      this config.
  """
    filter_arg = flag_config.add_mutually_exclusive_group()
    filter_arg.add_argument('--subscription-filter', default=None, help='CEL filter expression for the trigger. See https://cloud.google.com/build/docs/filter-build-events-using-cel for more details.\n')
    filter_arg.add_argument('--clear-subscription-filter', action='store_true', help='Clear existing subscription filter.')