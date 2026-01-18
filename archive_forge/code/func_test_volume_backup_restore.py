import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_backup_restore(self):
    """Test restore backup"""
    if not self.backup_enabled:
        self.skipTest('Backup service is not enabled')
    vol_id = uuid.uuid4().hex
    self.openstack('volume create ' + '--size 1 ' + vol_id, parse_output=True)
    self.wait_for_status('volume', vol_id, 'available')
    backup = self.openstack('volume backup create ' + vol_id, parse_output=True)
    self.wait_for_status('volume backup', backup['id'], 'available')
    backup_restored = self.openstack('volume backup restore %s %s' % (backup['id'], vol_id), parse_output=True)
    self.assertEqual(backup_restored['backup_id'], backup['id'])
    self.wait_for_status('volume backup', backup['id'], 'available')
    self.wait_for_status('volume', backup_restored['volume_id'], 'available')
    self.addCleanup(self.openstack, 'volume delete %s' % vol_id)