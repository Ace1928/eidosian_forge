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
def BuildPrometheusCondition(messages, group, rule):
    """Populates Alert Policy conditions translated from a Prometheus alert rule.

  Args:
    messages: Object containing information about all message types allowed.
    group: Information about the parent group of the current rule.
    rule: The current alert rule being translated into an Alert Policy.

  Raises:
    MissingRequiredFieldError: If the provided group/rule is missing an required
    field needed for translation.
    ValueError: If the provided rule name is not a valid Prometheus label name.

  Returns:
     The Alert Policy condition corresponding to the Prometheus group and rule
     provided.
  """
    condition = messages.Condition()
    condition.conditionPrometheusQueryLanguage = messages.PrometheusQueryLanguageCondition()
    if group.get('name') is None:
        raise MissingRequiredFieldError('Missing rule group name in field group.name')
    if rule.get('alert') is None:
        raise MissingRequiredFieldError('Missing alert rule name in field group.rules.alert')
    if _VALID_LABEL_REGEXP.fullmatch(rule.get('alert')) is None:
        raise ValueError('An invalid alert rule name in field group.rules.alert (not a valid PromQL label name)')
    if rule.get('expr') is None:
        raise MissingRequiredFieldError('Missing a PromQL expression in field groups.rules.expr')
    condition.conditionPrometheusQueryLanguage.ruleGroup = group.get('name')
    condition.displayName = rule.get('alert')
    condition.conditionPrometheusQueryLanguage.alertRule = rule.get('alert')
    condition.conditionPrometheusQueryLanguage.query = rule.get('expr')
    if rule.get('for') is not None:
        condition.conditionPrometheusQueryLanguage.duration = _FormatDuration(ConvertIntervalToSeconds(rule.get('for')))
    if group.get('interval') is not None:
        condition.conditionPrometheusQueryLanguage.evaluationInterval = ConvertPrometheusTimeStringToEvaluationDurationInSeconds(group.get('interval'))
    if rule.get('labels') is not None:
        condition.conditionPrometheusQueryLanguage.labels = messages.PrometheusQueryLanguageCondition.LabelsValue()
        for k, v in rule.get('labels').items():
            condition.conditionPrometheusQueryLanguage.labels.additionalProperties.append(messages.PrometheusQueryLanguageCondition.LabelsValue.AdditionalProperty(key=k, value=v))
    return condition