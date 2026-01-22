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
def AddLinuxSysctlFlags(parser, for_node_pool=False):
    """Adds Linux sysctl flag to the given parser."""
    if for_node_pool:
        help_text = 'Linux kernel parameters to be applied to all nodes in the new node pool as well\nas the pods running on the nodes.\n\nExamples:\n\n  $ {command} node-pool-1 --linux-sysctls="net.core.somaxconn=1024,net.ipv4.tcp_rmem=4096 87380 6291456"\n'
    else:
        help_text = 'Linux kernel parameters to be applied to all nodes in the new cluster\'s default\nnode pool as well as the pods running on the nodes.\n\nExamples:\n\n  $ {command} example-cluster --linux-sysctls="net.core.somaxconn=1024,net.ipv4.tcp_rmem=4096 87380 6291456"\n'
    parser.add_argument('--linux-sysctls', type=arg_parsers.ArgDict(min_length=1), default={}, help=help_text, metavar='KEY=VALUE', action=actions.DeprecationAction('--linux-sysctls', warn='The `--linux-sysctls` flag is deprecated. Please use `--system-config-from-file` instead. '))