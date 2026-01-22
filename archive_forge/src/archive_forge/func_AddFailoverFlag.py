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
def AddFailoverFlag(parser, show_negated_in_help=True):
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--failover', required=False, help='Whether the promote operation is a failover.', **kwargs)