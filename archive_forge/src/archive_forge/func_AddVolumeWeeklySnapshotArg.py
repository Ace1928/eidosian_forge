from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeWeeklySnapshotArg(parser):
    """Adds the --snapshot-weekly arg to the arg parser."""
    weekly_snapshot_arg_spec = {'snapshots-to-keep': float, 'minute': float, 'hour': float, 'day': str}
    weekly_snapshot_help = '\n  Make a snapshot every week e.g. at Monday 04:00, Wednesday 05:20,\n  Sunday 23:50\n  '
    parser.add_argument('--snapshot-weekly', type=arg_parsers.ArgDict(spec=weekly_snapshot_arg_spec), help=weekly_snapshot_help)