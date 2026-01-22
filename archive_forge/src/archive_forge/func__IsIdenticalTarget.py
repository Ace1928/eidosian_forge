from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
@classmethod
def _IsIdenticalTarget(cls, existing_exclusion, target_rule_set, target_rule_ids=None):
    return target_rule_set == existing_exclusion.targetRuleSet and set(target_rule_ids) == set(existing_exclusion.targetRuleIds)