from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_consistencygroup_description(self):
    v = cs.consistencygroups.list()[0]
    expected = {'consistencygroup': {'description': 'cg2 desc'}}
    vol = v.update(description='cg2 desc')
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)
    vol = cs.consistencygroups.update('1234', description='cg2 desc')
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)
    vol = cs.consistencygroups.update(v, description='cg2 desc')
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)