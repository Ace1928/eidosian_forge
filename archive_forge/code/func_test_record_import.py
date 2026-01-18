from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_record_import(self):
    backup_service = 'fake-backup-service'
    backup_url = 'fake-backup-url'
    expected_body = {'backup-record': {'backup_service': backup_service, 'backup_url': backup_url}}
    impt = cs.backups.import_record(backup_service, backup_url)
    cs.assert_called('POST', '/backups/import_record', expected_body)
    self._assert_request_id(impt)