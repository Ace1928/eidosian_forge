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
def AddLoggingVariantFlag(parser, for_node_pool=False, hidden=False):
    """Adds a --logging-variant flag to the given parser."""
    help_text = '  Specifies the logging variant that will be deployed on all the nodes\n  in the cluster. Valid logging variants are `MAX_THROUGHPUT`, `DEFAULT`.\n  If no value is specified, DEFAULT is used.'
    if for_node_pool:
        help_text = "        Specifies the logging variant that will be deployed on all the nodes\n        in the node pool. If the node pool doesn't specify a logging variant,\n        then the logging variant specified for the cluster will be deployed on\n        all the nodes in the node pool. Valid logging variants are\n        `MAX_THROUGHPUT`, `DEFAULT`."
    parser.add_argument('--logging-variant', help=help_text, hidden=hidden, choices={'DEFAULT': "                'DEFAULT' variant requests minimal resources but may not\n                guarantee high throughput. ", 'MAX_THROUGHPUT': "                'MAX_THROUGHPUT' variant requests more node resources and is\n                able to achieve logging throughput up to 10MB per sec. "}, default=None)