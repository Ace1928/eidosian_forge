from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_volume_unmanage(self):
    v = cs.volumes.get('1234')
    self._assert_request_id(v)
    vol = cs.volumes.unmanage(v)
    cs.assert_called('POST', '/volumes/1234/action', {'os-unmanage': None})
    self._assert_request_id(vol)