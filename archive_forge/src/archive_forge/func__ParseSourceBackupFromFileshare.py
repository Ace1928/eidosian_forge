from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.filestore.backups import util as backup_util
from googlecloudsdk.command_lib.filestore.snapshots import util as snapshot_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseSourceBackupFromFileshare(self, file_share):
    if 'source-backup' not in file_share:
        return None
    project = properties.VALUES.core.project.Get(required=True)
    location = file_share.get('source-backup-region')
    if location is None:
        raise InvalidArgumentError("If 'source-backup' is specified, 'source-backup-region' must also be specified.")
    return backup_util.BACKUP_NAME_TEMPLATE.format(project, location, file_share.get('source-backup'))