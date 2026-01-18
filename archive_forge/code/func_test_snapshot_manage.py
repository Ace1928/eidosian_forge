from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_snapshot_manage(self):
    vol = cs.volume_snapshots.manage('volume_id1', {'k': 'v'})
    expected = {'volume_id': 'volume_id1', 'name': None, 'description': None, 'metadata': None, 'ref': {'k': 'v'}}
    cs.assert_called('POST', '/os-snapshot-manage', {'snapshot': expected})
    self._assert_request_id(vol)