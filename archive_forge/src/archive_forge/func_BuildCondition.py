from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def BuildCondition(messages, condition=None, display_name=None, aggregations=None, trigger_count=None, trigger_percent=None, duration=None, condition_filter=None, if_value=None):
    """Populates the fields of a Condition message from args.

  Args:
    messages: module, module containing message classes for the stackdriver api
    condition: Condition or None, a base condition to populate the fields of.
    display_name: str, the display name for the condition.
    aggregations: list[Aggregation], list of Aggregation messages for the
      condition.
    trigger_count: int, corresponds to the count field of the condition trigger.
    trigger_percent: float, corresponds to the percent field of the condition
      trigger.
    duration: int, The amount of time that a time series must fail to report new
      data to be considered failing.
    condition_filter: str, A filter that identifies which time series should be
      compared with the threshold.
    if_value: tuple[str, float] or None, a tuple containing a string value
      corresponding to the comparison value enum and a float with the condition
      threshold value. None indicates that this should be an Absence condition.

  Returns:
    Condition, a condition with its fields populated from the args
  """
    if not condition:
        condition = messages.Condition()
    if display_name is not None:
        condition.displayName = display_name
    trigger = None
    if trigger_count or trigger_percent:
        trigger = messages.Trigger(count=trigger_count, percent=trigger_percent)
    kwargs = {'trigger': trigger, 'duration': duration, 'filter': condition_filter}
    if aggregations:
        kwargs['aggregations'] = aggregations
    if if_value is not None:
        comparator, threshold_value = if_value
        if not comparator:
            condition.conditionAbsent = messages.MetricAbsence(**kwargs)
        else:
            comparison_enum = messages.MetricThreshold.ComparisonValueValuesEnum
            condition.conditionThreshold = messages.MetricThreshold(comparison=getattr(comparison_enum, comparator), thresholdValue=threshold_value, **kwargs)
    return condition