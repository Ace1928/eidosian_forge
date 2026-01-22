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
def AddBinauthzFlags(parser, release_track=base.ReleaseTrack.GA, hidden=False, autopilot=False):
    """Adds Binary Authorization related flags to parser."""
    messages = apis.GetMessagesModule('container', api_adapter.APIVersionFromReleaseTrack(release_track))
    options = [api_adapter.NormalizeBinauthzEvaluationMode(option) for option in api_adapter.GetBinauthzEvaluationModeOptions(messages, release_track)]
    binauthz_group = parser.add_group(mutex=False, help='Flags for Binary Authorization:')
    if autopilot:
        binauthz_group.add_argument('--binauthz-evaluation-mode', choices=options, type=api_adapter.NormalizeBinauthzEvaluationMode, default=None, help='Enable Binary Authorization for this cluster.', hidden=hidden)
    else:
        binauthz_enablement_group = binauthz_group.add_group(mutex=True)
        binauthz_enablement_group.add_argument('--enable-binauthz', action=actions.DeprecationAction('--enable-binauthz', warn='The `--enable-binauthz` flag is deprecated. Please use `--binauthz-evaluation-mode` instead. ', action='store_true'), default=None, help='Enable Binary Authorization for this cluster.', hidden=hidden)
        binauthz_enablement_group.add_argument('--binauthz-evaluation-mode', choices=options, type=api_adapter.NormalizeBinauthzEvaluationMode, default=None, help='Enable Binary Authorization for this cluster.', hidden=hidden)
    if release_track in (base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA):
        platform_policy_type = arg_parsers.RegexpValidator(_BINAUTHZ_GKE_POLICY_REGEX, 'GKE policy resource names have the following format: `projects/{project_number}/platforms/gke/policies/{policy_id}`')
        binauthz_group.add_argument('--binauthz-policy-bindings', type=arg_parsers.ArgDict(spec={'name': platform_policy_type}, required_keys=['name'], max_length=1), metavar='name=BINAUTHZ_POLICY', action='append', default=None, help=textwrap.dedent('          The relative resource name of the Binary Authorization policy to audit\n          and/or enforce. GKE policies have the following format:\n          `projects/{project_number}/platforms/gke/policies/{policy_id}`.'), hidden=hidden)