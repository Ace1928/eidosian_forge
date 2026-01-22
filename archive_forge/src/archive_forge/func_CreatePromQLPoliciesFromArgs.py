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
def CreatePromQLPoliciesFromArgs(args, messages, channels=None):
    """Builds a PromQL policies message from args.

  Args:
    args: Flags provided by the user.
    messages: Object containing information about all message types allowed.
    channels: List of full Notification Channel names ("projects/<>/...") to be
      added to the translated policies.

  Returns:
     The Alert Policies corresponding to the Prometheus rules YAML file
     provided. In the case that no file is specified, the default behavior is to
     return an empty list.
  """
    if args.IsSpecified('policies_from_prometheus_alert_rules_yaml'):
        all_rule_yamls = args.policies_from_prometheus_alert_rules_yaml
        policies = []
        for rule_yaml in all_rule_yamls:
            policies += PrometheusMessageFromString(rule_yaml, messages, channels)
    else:
        policies = []
    return policies