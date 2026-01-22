from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.org_policies import exceptions
def _GetPolicyNameTokens(policy_name):
    """Returns the individual tokens from the policy name.

  Args:
    policy_name: The name of the policy. A policy name has the following syntax:
      [organizations|folders|projects]/{resource_id}/policies/{constraint_name}.
  """
    policy_name_tokens = policy_name.split('/')
    if len(policy_name_tokens) != 4:
        raise exceptions.InvalidInputError("Invalid policy name '{}': Name must be in the form [projects|folders|organizations]/{{resource_id}}/policies/{{constraint_name}}.".format(policy_name))
    return policy_name_tokens