from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetResponsePolicyRuleSpec(api_version):
    return concepts.ResourceSpec('dns.responsePolicyRules', api_version=api_version, resource_name='response_policy_rule', responsePolicy=ResponsePolicyAttributeConfig(), responsePolicyRule=ResponsePolicyRuleAttributeConfig(), project=ProjectAttributeConfig())