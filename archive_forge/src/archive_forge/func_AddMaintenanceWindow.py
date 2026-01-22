from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddMaintenanceWindow(parser, alloydb_messages, update=False):
    """Adds maintenance window related flags to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    alloydb_messages: Message module
    update: If false, only allow user to configure maintenance window
            day and hour.
  """
    if update:
        parent_group = parser.add_group(mutex=True, hidden=True, help='Configure a preferred maintenance window.')
        child_group = parent_group.add_group(help='Specify preferred day and time for maintenance.')
        _AddMaintenanceWindowAny(parent_group)
        _AddMaintenanceWindowDayAndHour(child_group, alloydb_messages)
    else:
        group = parser.add_group(hidden=True, help='Specify preferred day and time for maintenance.')
        _AddMaintenanceWindowDayAndHour(group, alloydb_messages)