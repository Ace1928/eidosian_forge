from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_role_assignments_inherited_list(self):
    ref_list = self.TEST_ALL_RESPONSE_LIST
    self.stub_entity('GET', [self.collection_key, '?scope.OS-INHERIT:inherited_to=projects'], entity=ref_list)
    returned_list = self.manager.list(os_inherit_extension_inherited_to='projects')
    self._assert_returned_list(ref_list, returned_list)
    query_string = 'scope.OS-INHERIT:inherited_to=projects'
    self.assertQueryStringIs(query_string)