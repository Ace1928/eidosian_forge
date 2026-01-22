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
def AddZonesPrimarySecondary(parser, help_text, hidden=False):
    """Adds the `--zone` and `--secondary-zone` to the parser."""
    zone_group = parser.add_group(required=False, hidden=hidden)
    zone_group.add_argument('--zone', required=False, help=help_text, hidden=hidden)
    zone_group.add_argument('--secondary-zone', required=False, help='Preferred secondary Compute Engine zone (e.g. us-central1-a, us-central1-b, etc.).', hidden=hidden)