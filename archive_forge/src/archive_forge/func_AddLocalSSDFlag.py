from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddLocalSSDFlag(parser, suppressed=False, help_text=''):
    """Adds a --local-ssd-count flag to the given parser."""
    help_text += 'The number of local SSD disks to provision on each node, formatted and mounted\nin the filesystem.\n\nLocal SSDs have a fixed 375 GB capacity per device. The number of disks that\ncan be attached to an instance is limited by the maximum number of disks\navailable on a machine, which differs by compute zone. See\nhttps://cloud.google.com/compute/docs/disks/local-ssd for more information.'
    parser.add_argument('--local-ssd-count', help=help_text, hidden=suppressed, type=int, default=0)