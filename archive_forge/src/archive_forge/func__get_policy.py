from keystone.common import sql
from keystone import exception
from keystone.policy.backends import rules
def _get_policy(self, session, policy_id):
    """Private method to get a policy model object (NOT a dictionary)."""
    ref = session.get(PolicyModel, policy_id)
    if not ref:
        raise exception.PolicyNotFound(policy_id=policy_id)
    return ref