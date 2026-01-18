from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_system_assignment_list(self):
    ref_list = self.TEST_USER_SYSTEM_LIST + self.TEST_GROUP_SYSTEM_LIST
    self.stub_entity('GET', [self.collection_key, '?scope.system=all'], entity=ref_list)
    returned_list = self.manager.list(system='all')
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'scope.system': 'all'}
    self.assertQueryStringContains(**kwargs)