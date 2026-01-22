from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_recovery_point_objective_flag(parser):
    """Adds the recovery point objective flag for buckets commands.

  Args:
    parser (parser_arguments.ArgumentInterceptor): Parser passed to surface.
  """
    parser.add_argument('--recovery-point-objective', '--rpo', choices=sorted([option.value for option in ReplicationStrategy]), metavar='SETTING', type=str, help='Sets the [recovery point objective](https://cloud.google.com/architecture/dr-scenarios-planning-guide#basics_of_dr_planning) of a bucket. This flag can only be used with multi-region and dual-region buckets. `DEFAULT` option is valid for multi-region and dual-regions buckets. `ASYNC_TURBO` option is only valid for dual-region buckets. If unspecified when the bucket is created, it defaults to `DEFAULT` for dual-region and multi-region buckets. For more information, see [replication in Cloud Storage](https://cloud.google.com/storage/docs/availability-durability#cross-region-redundancy).')