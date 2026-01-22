from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def ParseAccount(account_id):
    return GetRegistry().Parse(account_id, collection=ACCOUNTS_COLLECTION)