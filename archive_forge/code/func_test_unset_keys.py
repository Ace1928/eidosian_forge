from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import group_types
def test_unset_keys(self):
    t = cs.group_types.get(1)
    res = t.unset_keys(['k'])
    cs.assert_called('DELETE', '/group_types/1/group_specs/k')
    self._assert_request_id(res)