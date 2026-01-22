from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_client_types(self, object):
    elements = object.findall(fixxpath('backupClientType', BACKUP_NS))
    return [self._to_client_type(el) for el in elements]