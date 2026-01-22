from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def RemovePolicyBindingFromCryptoKey(crypto_key_ref, member, role):
    """Does an atomic Read-Modify-Write, removing the member from the role."""
    policy = GetCryptoKeyIamPolicy(crypto_key_ref)
    policy.version = iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION
    iam_util.RemoveBindingFromIamPolicy(policy, member, role)
    return SetCryptoKeyIamPolicy(crypto_key_ref, policy, update_mask='bindings,etag')