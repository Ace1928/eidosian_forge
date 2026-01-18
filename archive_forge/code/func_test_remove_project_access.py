from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_type_access
def test_remove_project_access(self):
    access = cs.volume_type_access.remove_project_access('3', PROJECT_UUID)
    cs.assert_called('POST', '/types/3/action', {'removeProjectAccess': {'project': PROJECT_UUID}})
    self._assert_request_id(access)