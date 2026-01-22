from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaSearchApplicablePoliciesResponseBindingAndPolicy(_messages.Message):
    """A pair of a binding and a policy referenced by that binding (if
  accessible)

  Fields:
    binding: A binding between a target and a policy
    policy: The policy associated with the above binding. Omitted if the
      policy cannot be retrieved due to lack of permissions
    policyInaccessible: Will be set if there was a permission error getting
      the policy (even though the binding was accessible).
  """
    binding = _messages.MessageField('GoogleIamV3betaPolicyBinding', 1)
    policy = _messages.MessageField('GoogleIamV3betaPolicy', 2)
    policyInaccessible = _messages.MessageField('GoogleIamV3betaPolicyInaccessible', 3)