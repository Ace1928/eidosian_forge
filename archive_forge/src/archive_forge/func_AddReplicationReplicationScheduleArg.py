from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddReplicationReplicationScheduleArg(parser, required=True):
    """Adds the Replication Schedule (--replication-schedule) arg to the given parser.

  Args:
    parser: Argparse parser.
    required: If the Replication Schedule is required. E.g. it is required by
      Create, but not required by Update.
  """
    parser.add_argument('--replication-schedule', type=str, required=required, help='The schedule for the Replication.')