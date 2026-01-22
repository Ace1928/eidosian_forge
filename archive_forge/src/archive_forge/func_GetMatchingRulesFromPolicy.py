from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.org_policies import exceptions
def GetMatchingRulesFromPolicy(policy, condition_expression=None):
    """Returns a list of rules on the policy that contain the specified condition expression.

  In the case that condition_expression is None, rules without conditions are
  returned.

  Args:
    policy: messages.GoogleCloudOrgpolicy{api_version}Policy, The policy object
      to search.
    condition_expression: str, The condition expression to look for.
  """
    if condition_expression is None:
        condition_filter = lambda rule: rule.condition is None
    else:
        condition_filter = lambda rule: rule.condition is not None and rule.condition.expression == condition_expression
    return list(filter(condition_filter, policy.spec.rules))