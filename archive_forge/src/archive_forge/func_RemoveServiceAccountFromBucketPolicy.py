from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.command_lib.iam import iam_util
def RemoveServiceAccountFromBucketPolicy(bucket_ref, member, role):
    """Deauthorize Account for Buckets."""
    policy = storage_api.StorageClient().GetIamPolicy(bucket_ref)
    iam_util.RemoveBindingFromIamPolicy(policy, member, role)
    return storage_api.StorageClient().SetIamPolicy(bucket_ref, policy)