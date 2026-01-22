from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddCreateWorkloadFlags(parser, release_track):
    """Adds required flags to the assured workloads create command.

  Args:
    parser: Parser, Parser used to construct the command flags.
    release_track: ReleaseTrack, Release track of the command being called.

  Returns:
    None.
  """
    parser.add_argument('--location', required=True, help='The location of the new Assured Workloads environment. For a current list of supported LOCATION values, see [Assured Workloads locations](https://cloud.google.com/assured-workloads/docs/locations).')
    parser.add_argument('--organization', required=True, help='The parent organization of the new Assured Workloads environment, provided as an organization ID')
    parser.add_argument('--external-identifier', help='The external identifier of the new Assured Workloads environment')
    parser.add_argument('--display-name', required=True, help='The display name of the new Assured Workloads environment')
    parser.add_argument('--compliance-regime', required=True, choices=compliance_regimes.get(release_track), help='The compliance regime of the new Assured Workloads environment')
    parser.add_argument('--partner', choices=PARTNERS, help='The partner choice when creating a workload managed by local trusted partners.')
    parser.add_argument('--partner-permissions', type=arg_parsers.ArgDict(spec={'data-logs-viewer': bool}), metavar='KEY=VALUE', help='The partner permissions for the partner regime, for example, data-logs-viewer=true/false')
    parser.add_argument('--billing-account', required=True, help='The billing account of the new Assured Workloads environment, for example, billingAccounts/0000AA-AAA00A-A0A0A0')
    parser.add_argument('--next-rotation-time', help='The next rotation time of the KMS settings of new Assured Workloads environment, for example, 2020-12-30T10:15:30.00Z')
    parser.add_argument('--rotation-period', help='The rotation period of the KMS settings of the new Assured Workloads environment, for example, 172800s')
    parser.add_argument('--labels', type=arg_parsers.ArgDict(), metavar='KEY=VALUE', help='The labels of the new Assured Workloads environment, for example, LabelKey1=LabelValue1,LabelKey2=LabelValue2')
    parser.add_argument('--provisioned-resources-parent', help='The parent of the provisioned projects, for example, folders/{FOLDER_ID}')
    parser.add_argument('--enable-sovereign-controls', type=bool, default=False, help='If true, enable sovereign controls for the new Assured Workloads environment, currently only supported by EU_REGIONS_AND_SUPPORT')
    _AddResourceSettingsFlag(parser, release_track)