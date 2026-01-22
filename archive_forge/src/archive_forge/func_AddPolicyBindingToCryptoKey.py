from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def AddPolicyBindingToCryptoKey(crypto_key_ref, member, role):
    """Add an IAM policy binding on the CryptoKey.

  Does an atomic Read-Modify-Write, adding the member to the role.

  Args:
    crypto_key_ref: A resources.Resource naming the CryptoKey.
    member: Principal to add to the policy binding.
    role: List of roles to add to the policy binding.

  Returns:
    The new IAM Policy.
  """
    return AddPolicyBindingsToCryptoKey(crypto_key_ref, [(member, role)])