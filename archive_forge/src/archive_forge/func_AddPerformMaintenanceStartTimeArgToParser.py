from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPerformMaintenanceStartTimeArgToParser(parser):
    """Add --start-time flag."""
    parser.add_argument('--start-time', metavar='START_TIME', type=arg_parsers.Datetime.Parse, help='The requested time for the maintenance window to start. The timestamp must be an RFC3339 valid string.')