from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_backup_job(self, element, target, client_id):
    running_job = element.find(fixxpath('runningJob', BACKUP_NS))
    if running_job is not None:
        return BackupTargetJob(id=running_job.get('id'), status=running_job.get('status'), progress=int(running_job.get('percentageComplete')), driver=self.connection.driver, target=target, extra={'clientId': client_id})
    return None