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
def AddEnableMasterSignalsFlags(parser, for_create=False):
    """Adds --master-logs and --enable-master-metrics flags to parser."""
    help_text = 'Set which master components logs should be sent to Cloud Operations.\n\nExamples:\n\n  $ {command} --master-logs APISERVER,SCHEDULER\n'
    if for_create:
        group = parser.add_group(hidden=True)
    else:
        group = parser.add_mutually_exclusive_group(hidden=True)
    group.add_argument('--master-logs', type=arg_parsers.ArgList(choices=api_adapter.PRIMARY_LOGS_OPTIONS), help=help_text, metavar='COMPONENT', action=actions.DeprecationAction('--master-logs', warn='The `--master-logs` flag is deprecated and will be removed in an upcoming release. Please use `--logging` instead. For more information, please read: https://cloud.google.com/stackdriver/docs/solutions/gke/installing.'))
    if not for_create:
        help_text = 'Disable sending logs from master components to Cloud Operations.\n'
        group.add_argument('--no-master-logs', action=actions.DeprecationAction('--no-master-logs', warn='The `--no-master-logs` flag is deprecated and will be removed in an upcoming release. Please use `--logging` instead. For more information, please read: https://cloud.google.com/stackdriver/docs/solutions/gke/installing.', action='store_true'), default=False, help=help_text)
    help_text = 'Enable sending metrics from master components to Cloud Operations.\n'
    group.add_argument('--enable-master-metrics', action=actions.DeprecationAction('--enable-master-metrics', warn='The `--enable-master-metrics` flag is deprecated and will be removed in an upcoming release. Please use `--monitoring` instead. For more information, please read: https://cloud.google.com/stackdriver/docs/solutions/gke/installing.', action='store_true'), default=None, help=help_text)