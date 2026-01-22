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
def AddBakExportDifferentialBaseArgument(parser):
    """Add the 'dfferential-base' argument to the parser for export."""
    parser.add_argument('--differential-base', required=False, default=False, action='store_true', help='Whether the bak file export can be used as differential base for future differential backup. SQL Server only')