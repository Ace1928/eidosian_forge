from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_role_assignments_list(self):
    ref_list = self.TEST_ALL_RESPONSE_LIST
    self.stub_entity('GET', [self.collection_key, '?role.id=' + self.TEST_ROLE_ID], entity=ref_list)
    returned_list = self.manager.list(role=self.TEST_ROLE_ID)
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'role.id': self.TEST_ROLE_ID}
    self.assertQueryStringContains(**kwargs)