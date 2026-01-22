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
def AddOffloadArgument(parser):
    """Add the 'offload' argument to the parser."""
    parser.add_argument('--offload', action='store_true', help='Offload an export to a temporary instance. Doing so reduces strain on source instances and allows other operations to be performed while the export is in progress.')