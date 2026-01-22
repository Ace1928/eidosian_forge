from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.util import times
def ModifyCreateRequest(backup_ref, args, req):
    """Parse argument and construct create backup request."""
    req.backup.sourceTable = 'projects/{0}/instances/{1}/tables/{2}'.format(backup_ref.projectsId, backup_ref.instancesId, args.table)
    req.backup.expireTime = GetExpireTime(args)
    req.backupId = args.backup
    req.parent = backup_ref.Parent().RelativeName()
    return req