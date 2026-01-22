from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddLicenceBasedFlags(parser):
    parser.add_argument('--license', required=True, help='Applicable license URI. For example: `https://www.googleapis.com/compute/v1/projects/suse-sap-cloud/global/licenses/sles-sap-12`')
    parser.add_argument('--cores-per-license', required=False, type=str, help='Core range of the instance. Must be one of: `1-2`, `3-4`, `5+`. Required for SAP licenses.')
    parser.add_argument('--amount', required=True, type=int, help='Number of licenses purchased.')
    AddPlanForCreate(parser)