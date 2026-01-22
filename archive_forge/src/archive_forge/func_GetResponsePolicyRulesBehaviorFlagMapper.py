from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetResponsePolicyRulesBehaviorFlagMapper(messages):
    return arg_utils.ChoiceEnumMapper('--behavior', messages.ResponsePolicyRule.BehaviorValueValuesEnum, include_filter=lambda x: x != 'behaviorUnspecified', help_str='The response policy rule query behavior.')