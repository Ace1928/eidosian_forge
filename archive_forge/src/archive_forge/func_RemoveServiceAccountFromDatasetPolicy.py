from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.command_lib.iam import iam_util
def RemoveServiceAccountFromDatasetPolicy(dataset_policy, member, role):
    """Deauthorize Account for Dataset."""
    for entry in dataset_policy.access:
        if entry.role == role and member in entry.userByEmail:
            dataset_policy.access.remove(entry)
            return True
    return False