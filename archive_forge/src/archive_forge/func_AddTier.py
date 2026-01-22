from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddTier(parser, is_patch=False, hidden=False):
    """Adds '--tier' flag to the parser."""
    help_text = "Machine type for a shared-core instance e.g. ``db-g1-small''. For all other instances, instead of using tiers, customize your instance by specifying its CPU and memory. You can do so with the `--cpu` and `--memory` flags. Learn more about how CPU and memory affects pricing: https://cloud.google.com/sql/pricing."
    if is_patch:
        help_text += ' WARNING: Instance will be restarted.'
    parser.add_argument('--tier', '-t', required=False, help=help_text, hidden=hidden)