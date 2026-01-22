from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
def _test_absolute_limit(self, expected, actual):
    self.assertEqual(expected['totalSnapshotsUsed'], actual.total_snapshots_used)
    self.assertEqual(expected['maxTotalBackups'], actual.max_total_backups)
    self.assertEqual(expected['maxTotalVolumeGigabytes'], actual.max_total_volume_gigabytes)
    self.assertEqual(expected['maxTotalSnapshots'], actual.max_total_snapshots)
    self.assertEqual(expected['maxTotalBackupGigabytes'], actual.max_total_backup_gigabytes)
    self.assertEqual(expected['totalBackupGigabytesUsed'], actual.total_backup_gigabytes_used)
    self.assertEqual(expected['maxTotalVolumes'], actual.max_total_volumes)
    self.assertEqual(expected['totalVolumesUsed'], actual.total_volumes_used)
    self.assertEqual(expected['totalBackupsUsed'], actual.total_backups_used)
    self.assertEqual(expected['totalGigabytesUsed'], actual.total_gigabytes_used)