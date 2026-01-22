from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_alert(self, element):
    alert = element.find(fixxpath('alerting', BACKUP_NS))
    if alert is not None:
        notify_list = [email_addr.text for email_addr in alert.findall(fixxpath('emailAddress', BACKUP_NS))]
        return DimensionDataBackupClientAlert(trigger=element.get('trigger'), notify_list=notify_list)
    return None