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
def AddShowEdition(parser):
    """Show the instance or tier edition."""
    kwargs = _GetKwargsForBoolFlag(False)
    parser.add_argument('--show-edition', required=False, help='Show the edition field.', **kwargs)