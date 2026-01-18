from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_project_assignments_list_include_subtree(self):
    ref_list = self.TEST_GROUP_PROJECT_LIST + self.TEST_USER_PROJECT_LIST
    self.stub_entity('GET', [self.collection_key, '?scope.project.id=%s&include_subtree=True' % self.TEST_TENANT_ID], entity=ref_list)
    returned_list = self.manager.list(project=self.TEST_TENANT_ID, include_subtree=True)
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'scope.project.id': self.TEST_TENANT_ID, 'include_subtree': 'True'}
    self.assertQueryStringContains(**kwargs)