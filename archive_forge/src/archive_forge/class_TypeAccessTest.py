from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_type_access
class TypeAccessTest(utils.TestCase):

    def test_list(self):
        access = cs.volume_type_access.list(volume_type='3')
        cs.assert_called('GET', '/types/3/os-volume-type-access')
        self._assert_request_id(access)
        for a in access:
            self.assertIsInstance(a, volume_type_access.VolumeTypeAccess)

    def test_add_project_access(self):
        access = cs.volume_type_access.add_project_access('3', PROJECT_UUID)
        cs.assert_called('POST', '/types/3/action', {'addProjectAccess': {'project': PROJECT_UUID}})
        self._assert_request_id(access)

    def test_remove_project_access(self):
        access = cs.volume_type_access.remove_project_access('3', PROJECT_UUID)
        cs.assert_called('POST', '/types/3/action', {'removeProjectAccess': {'project': PROJECT_UUID}})
        self._assert_request_id(access)