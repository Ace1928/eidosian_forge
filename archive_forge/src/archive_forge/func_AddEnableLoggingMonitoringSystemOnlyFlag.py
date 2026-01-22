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
def AddEnableLoggingMonitoringSystemOnlyFlag(parser):
    """Adds a --enable-stackdriver-kubernetes-system flag to parser."""
    help_text = 'Enable Cloud Operations system-only monitoring and logging.'
    parser.add_argument('--enable-logging-monitoring-system-only', action=actions.DeprecationAction('--enable-logging-monitoring-system-only', warn='The `--enable-logging-monitoring-system-only` flag is deprecated and will be removed in an upcoming release. Please use `--logging` and `--monitoring` instead. For more information, please read: https://cloud.google.com/stackdriver/docs/solutions/gke/installing.', action='store_true'), help=help_text)