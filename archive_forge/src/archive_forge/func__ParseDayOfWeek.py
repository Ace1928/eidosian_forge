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
def _ParseDayOfWeek(value):
    value_upper = value.upper()
    if value_upper not in visible_choices_set:
        raise arg_parsers.ArgumentTypeError('{value} must be one of [{choices}]'.format(value=value, choices=', '.join(visible_choices)))
    return day_of_week_enum.lookup_by_name(value_upper)