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
def _GetDayOfWeekArgList(alloydb_messages):
    """Returns an ArgList accepting days of the week."""
    day_of_week_enum = alloydb_messages.WeeklySchedule.DaysOfWeekValueListEntryValuesEnum
    choices = [day_of_week_enum.lookup_by_number(i) for i in range(1, 8)]
    visible_choices = [c.name for c in choices]
    visible_choices_set = set(visible_choices)

    def _ParseDayOfWeek(value):
        value_upper = value.upper()
        if value_upper not in visible_choices_set:
            raise arg_parsers.ArgumentTypeError('{value} must be one of [{choices}]'.format(value=value, choices=', '.join(visible_choices)))
        return day_of_week_enum.lookup_by_name(value_upper)
    return arg_parsers.ArgList(element_type=_ParseDayOfWeek, choices=choices, visible_choices=visible_choices)