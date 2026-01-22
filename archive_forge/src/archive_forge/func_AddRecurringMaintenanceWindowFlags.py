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
def AddRecurringMaintenanceWindowFlags(parser, hidden=False, is_update=False):
    """Adds flags related to recurring maintenance windows to the parser."""
    hidden_for_create = hidden and (not is_update)
    if is_update:
        group = parser.add_group(hidden=hidden, mutex=True)
    else:
        group = parser
    set_window_group = group.add_group(hidden=hidden_for_create, help="Set a flexible maintenance window by specifying a window that recurs per an\nRFC 5545 RRULE. Non-emergency maintenance will occur in the recurring windows.\n+\nExamples:\n+\nFor a 9-5 Mon-Wed UTC-4 maintenance window:\n+\n  $ {command} example-cluster --maintenance-window-start=2000-01-01T09:00:00-04:00 --maintenance-window-end=2000-01-01T17:00:00-04:00 --maintenance-window-recurrence='FREQ=WEEKLY;BYDAY=MO,TU,WE'\n+\nFor a daily window from 22:00 - 04:00 UTC:\n+\n  $ {command} example-cluster --maintenance-window-start=2000-01-01T22:00:00Z --maintenance-window-end=2000-01-02T04:00:00Z --maintenance-window-recurrence=FREQ=DAILY\n")
    set_window_group.add_argument('--maintenance-window-start', type=arg_parsers.Datetime.Parse, required=True, hidden=hidden_for_create, metavar='TIME_STAMP', help='Start time of the first window (can occur in the past). The start time\ninfluences when the window will start for recurrences. See $ gcloud topic\ndatetimes for information on time formats.\n')
    set_window_group.add_argument('--maintenance-window-end', type=arg_parsers.Datetime.Parse, required=True, hidden=hidden_for_create, metavar='TIME_STAMP', help='End time of the first window (can occur in the past). Must take place after the\nstart time. The difference in start and end time specifies the length of each\nrecurrence. See $ gcloud topic datetimes for information on time formats.\n')
    set_window_group.add_argument('--maintenance-window-recurrence', type=str, required=True, hidden=hidden_for_create, metavar='RRULE', help='An RFC 5545 RRULE, specifying how the window will recur. Note that minimum\nrequirements for maintenance periods will be enforced. Note that FREQ=SECONDLY,\nMINUTELY, and HOURLY are not supported.\n')
    if is_update:
        group.add_argument('--clear-maintenance-window', action='store_true', default=False, help='If set, remove the maintenance window that was set with --maintenance-window\nfamily of flags.\n')
        AddMaintenanceExclusionFlags(group)