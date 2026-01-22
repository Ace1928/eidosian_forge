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
def CreateAlertPolicyFromArgs(args, messages):
    """Builds an AleryPolicy message from args."""
    policy_base_flags = ['--display-name', '--policy', '--policy-from-file']
    ValidateAtleastOneSpecified(args, policy_base_flags)
    policy = GetBasePolicyMessageFromArgs(args, messages.AlertPolicy)
    combiner = args.combiner if args.IsSpecified('combiner') else None
    enabled = args.enabled if args.IsSpecified('enabled') else None
    channel_refs = args.CONCEPTS.notification_channels.Parse() or []
    channels = [channel.RelativeName() for channel in channel_refs] or None
    documentation_content = args.documentation or args.documentation_from_file
    documentation_format = args.documentation_format if documentation_content else None
    ModifyAlertPolicy(policy, messages, display_name=args.display_name, combiner=combiner, documentation_content=documentation_content, documentation_format=documentation_format, enabled=enabled, channels=channels)
    if CheckConditionArgs(args):
        aggregations = None
        if args.aggregation:
            aggregations = [MessageFromString(args.aggregation, messages.Aggregation, 'Aggregation')]
        condition = BuildCondition(messages, display_name=args.condition_display_name, aggregations=aggregations, trigger_count=args.trigger_count, trigger_percent=args.trigger_percent, duration=_FormatDuration(args.duration), condition_filter=args.condition_filter, if_value=args.if_value)
        policy.conditions.append(condition)
    return policy