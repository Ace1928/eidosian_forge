from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_system_assignment_list_for_user(self):
    ref_list = self.TEST_USER_SYSTEM_LIST
    self.stub_entity('GET', [self.collection_key, '?user.id=%s&scope.system=all' % self.TEST_USER_ID], entity=ref_list)
    returned_list = self.manager.list(system='all', user=self.TEST_USER_ID)
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'scope.system': 'all', 'user.id': self.TEST_USER_ID}
    self.assertQueryStringContains(**kwargs)