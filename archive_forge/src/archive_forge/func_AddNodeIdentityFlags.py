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
def AddNodeIdentityFlags(parser, example_target):
    """Adds node identity flags to the given parser.

  Node identity flags are --scopes and --service-account.

  Args:
    parser: A given parser.
    example_target: the target for the command, e.g. mycluster.
  """
    node_identity_group = parser.add_group(help='Options to specify the node identity.')
    scopes_group = node_identity_group.add_group(help='Scopes options.')
    scopes_group.add_argument('--scopes', type=arg_parsers.ArgList(), metavar='SCOPE', default='gke-default', help='Specifies scopes for the node instances.\n\nExamples:\n\n  $ {{command}} {example_target} --scopes=https://www.googleapis.com/auth/devstorage.read_only\n\n  $ {{command}} {example_target} --scopes=bigquery,storage-rw,compute-ro\n\nMultiple scopes can be specified, separated by commas. Various scopes are\nautomatically added based on feature usage. Such scopes are not added if an\nequivalent scope already exists.\n\n- `monitoring-write`: always added to ensure metrics can be written\n- `logging-write`: added if Cloud Logging is enabled\n  (`--enable-cloud-logging`/`--logging`)\n- `monitoring`: added if Cloud Monitoring is enabled\n  (`--enable-cloud-monitoring`/`--monitoring`)\n- `gke-default`: added for Autopilot clusters that use the default service\n  account\n- `cloud-platform`: added for Autopilot clusters that use any other service\n  account\n\n{scopes_help}\n'.format(example_target=example_target, scopes_help=compute_constants.ScopesHelp()))
    sa_help_text = 'The Google Cloud Platform Service Account to be used by the node VMs. If a service account is specified, the cloud-platform and userinfo.email scopes are used. If no Service Account is specified, the project default service account is used.'
    node_identity_group.add_argument('--service-account', help=sa_help_text)