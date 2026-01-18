from cinderclient.tests.unit.fixture_data import client
from cinderclient.tests.unit.fixture_data import snapshots
from cinderclient.tests.unit import utils
def test_update_snapshot_status_with_progress(self):
    s = self.cs.volume_snapshots.get('1234')
    self._assert_request_id(s)
    stat = {'status': 'available', 'progress': '73%'}
    stats = self.cs.volume_snapshots.update_snapshot_status(s, stat)
    self.assert_called('POST', '/snapshots/1234/action')
    self._assert_request_id(stats)