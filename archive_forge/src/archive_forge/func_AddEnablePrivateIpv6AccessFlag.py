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
def AddEnablePrivateIpv6AccessFlag(parser, hidden=False):
    """Adds --enable-private-ipv6-access flag to the parser.

  When enabled, this allows gRPC clients on this cluster's pods a fast
  path to access Google hosted services (eg. Cloud Spanner,
  Cloud Dataflow, Cloud Bigtable)
  This is currently only available on Alpha clusters, and needs
  '--enable-kubernetes-alpha' to be specified also.

  Args:
    parser: A given parser.
    hidden: If true, suppress help text for added options.
  """
    parser.add_argument('--enable-private-ipv6-access', default=None, help="Enables private access to Google services over IPv6.\n\nWhen enabled, this allows gRPC clients on this cluster's pods a fast path to\naccess Google hosted services (eg. Cloud Spanner, Cloud Dataflow, Cloud\nBigtable).\n\nThis is currently only available on Alpha clusters, specified by using\n--enable-kubernetes-alpha.\n      ", hidden=hidden, action='store_true')