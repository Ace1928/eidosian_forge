from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _BucketAllowsPublicObjectReads(bucket):
    return any([acl.entity.lower() == 'allusers' and acl.role.lower() == 'reader' for acl in bucket.defaultObjectAcl])