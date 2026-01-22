from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
def IsOpsAgentPolicy(guest_policy):
    """Validate whether an OS Conifg guest policy is an Ops Agent Policy.

  Args:
    guest_policy: Client message of OS Config guest policy.


  Returns:
    True if it is an Ops Agent Policy type OS Config guest policy.
  """
    if guest_policy.description is None:
        return False
    try:
        guest_policy_description = json.loads(guest_policy.description)
    except ValueError:
        return False
    return isinstance(guest_policy_description, dict) and 'type' in guest_policy_description and (guest_policy_description['type'] == _GUEST_POLICY_TYPE_OPS_AGENT)