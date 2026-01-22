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
def AddManagedPrometheusFlags(parser, for_create=False):
    """Adds --enable-managed-prometheus and --disable-managed-prometheus flags to parser."""
    enable_help_text = '\n  Enables managed collection for Managed Service for Prometheus in the cluster.\n\n  See https://cloud.google.com/stackdriver/docs/managed-prometheus/setup-managed#enable-mgdcoll-gke\n  for more info.\n\n  Enabled by default for cluster versions 1.27 or greater,\n  use --no-enable-managed-prometheus to disable.\n  '
    disable_help_text = 'Disable managed collection for Managed Service for\n  Prometheus.'
    if for_create:
        parser.add_argument('--enable-managed-prometheus', action='store_true', default=None, help=enable_help_text)
    else:
        group = parser.add_group(mutex=True)
        group.add_argument('--enable-managed-prometheus', action='store_true', default=None, help=enable_help_text)
        group.add_argument('--disable-managed-prometheus', action='store_true', default=None, help=disable_help_text)