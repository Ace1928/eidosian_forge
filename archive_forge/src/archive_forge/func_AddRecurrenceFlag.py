from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
def AddRecurrenceFlag(parser):
    """Adds flag for recurrence to the given parser.

  Args:
    parser: The argparse parser.
  """
    group = parser.add_group(help='Recurrence settings of a backup schedule.', required=True)
    help_text = '      The recurrence settings of a backup schedule.\n\n      Currently only daily and weekly backup schedules are supported.\n\n      When a weekly backup schedule is created, day-of-week is needed.\n\n      For example, to create a weekly backup schedule which creates backups on\n      Monday.\n\n        $ {command} --recurrence=weekly --day-of-week=MON\n  '
    group.add_argument('--recurrence', type=str, help=help_text, required=True)
    help_text = '     The day of week (UTC time zone) of when backups are created.\n\n      The available values are: `MON`, `TUE`, `WED`, `THU`, `FRI`, `SAT`,`SUN`.\n      Values are case insensitive.\n\n      This is required when creating a weekly backup schedule.\n  '
    group.add_argument('--day-of-week', choices=arg_parsers.DayOfWeek.DAYS, type=arg_parsers.DayOfWeek.Parse, help=help_text, required=False)