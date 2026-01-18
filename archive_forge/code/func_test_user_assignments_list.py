from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_user_assignments_list(self):
    ref_list = self.TEST_USER_DOMAIN_LIST + self.TEST_USER_PROJECT_LIST
    self.stub_entity('GET', [self.collection_key, '?user.id=%s' % self.TEST_USER_ID], entity=ref_list)
    returned_list = self.manager.list(user=self.TEST_USER_ID)
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'user.id': self.TEST_USER_ID}
    self.assertQueryStringContains(**kwargs)