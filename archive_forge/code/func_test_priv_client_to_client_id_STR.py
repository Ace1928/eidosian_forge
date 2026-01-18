import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.backup.base import BackupTargetJob
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import BackupFileFixtures
from libcloud.common.dimensiondata import DimensionDataAPIException
from libcloud.backup.drivers.dimensiondata import DEFAULT_BACKUP_PLAN
from libcloud.backup.drivers.dimensiondata import DimensionDataBackupDriver as DimensionData
def test_priv_client_to_client_id_STR(self):
    self.assertEqual(self.driver._client_to_client_id('30b1ff76-c76d-4d7c-b39d-3b72be0384c8'), '30b1ff76-c76d-4d7c-b39d-3b72be0384c8')