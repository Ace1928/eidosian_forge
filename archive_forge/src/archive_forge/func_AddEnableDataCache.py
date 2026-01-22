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
def AddEnableDataCache(parser, show_negated_in_help=False, hidden=False):
    """Adds '--enable-data-cache' flag to the parser."""
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--enable-data-cache', required=False, help='Enable use of data cache for accelerated read performance. This flag is only available for Enterprise_Plus edition instances.', hidden=hidden, **kwargs)