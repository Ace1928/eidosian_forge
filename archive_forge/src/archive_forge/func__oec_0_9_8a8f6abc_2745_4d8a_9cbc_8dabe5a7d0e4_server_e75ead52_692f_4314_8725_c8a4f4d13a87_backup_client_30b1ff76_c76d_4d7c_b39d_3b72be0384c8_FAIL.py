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
def _oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_e75ead52_692f_4314_8725_c8a4f4d13a87_backup_client_30b1ff76_c76d_4d7c_b39d_3b72be0384c8_FAIL(self, method, url, body, headers):
    if url.endswith('disable'):
        body = self.fixtures.load('_remove_backup_client_FAIL.xml')
    elif url.endswith('cancelJob'):
        body = self.fixtures.load('_backup_client_30b1ff76_c76d_4d7c_b39d_3b72be0384c8_cancelJob_FAIL.xml')
    else:
        raise ValueError('Unknown URL: %s' % url)
    return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.OK])