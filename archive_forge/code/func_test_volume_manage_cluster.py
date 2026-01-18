from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_volume_manage_cluster(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.16'))
    vol = cs.volumes.manage(None, {'k': 'v'}, cluster='cluster1')
    expected = {'host': None, 'name': None, 'availability_zone': None, 'description': None, 'metadata': None, 'ref': {'k': 'v'}, 'volume_type': None, 'bootable': False, 'cluster': 'cluster1'}
    cs.assert_called('POST', '/os-volume-manage', {'volume': expected})
    self._assert_request_id(vol)