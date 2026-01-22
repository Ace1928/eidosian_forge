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
def AddAllowRouteOverlapFlag(parser):
    """Adds a --allow-route-overlap flag to parser."""
    help_text = "Allows the provided cluster CIDRs to overlap with existing routes\nthat are less specific and do not terminate at a VM.\n\nWhen enabled, `--cluster-ipv4-cidr` must be fully specified (e.g. `10.96.0.0/14`\n, but not `/14`). If `--enable-ip-alias` is also specified, both\n`--cluster-ipv4-cidr` and `--services-ipv4-cidr` must be fully specified.\n\nMust be used in conjunction with '--enable-ip-alias' or '--no-enable-ip-alias'.\n"
    parser.add_argument('--allow-route-overlap', action='store_true', default=None, help=help_text)