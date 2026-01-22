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
def AddEnableAutoUpgradeFlag(parser, for_node_pool=False, suppressed=False, default=None):
    """Adds a --enable-autoupgrade flag to parser."""
    if for_node_pool:
        help_text = 'Sets autoupgrade feature for a node pool.\n\n  $ {command} node-pool-1 --cluster=example-cluster --enable-autoupgrade\n'
    else:
        help_text = "Sets autoupgrade feature for a cluster's default node pool(s).\n\n  $ {command} example-cluster --enable-autoupgrade\n"
    help_text += '\nSee https://cloud.google.com/kubernetes-engine/docs/node-auto-upgrades for more info.'
    parser.add_argument('--enable-autoupgrade', action='store_true', default=default, help=help_text, hidden=suppressed)