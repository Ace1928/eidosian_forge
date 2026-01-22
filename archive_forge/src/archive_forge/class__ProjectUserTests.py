import copy
import http.client
import uuid
from oslo_serialization import jsonutils
from keystone.common.policies import role_assignment as rp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _ProjectUserTests(object):

    def test_user_cannot_list_all_assignments_in_their_project(self):
        with self.test_client() as c:
            c.get('/v3/role_assignments', headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_role_assignments_by_user_of_project(self):
        assignments = self._setup_test_role_assignments()
        user_id = assignments['user_id']
        with self.test_client() as c:
            c.get('/v3/role_assignments?user.id=%s' % user_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_role_assignments_by_group_of_project(self):
        assignments = self._setup_test_role_assignments()
        group_id = assignments['group_id']
        with self.test_client() as c:
            c.get('/v3/role_assignments?group.id=%s' % group_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_role_assignments_by_system(self):
        with self.test_client() as c:
            c.get('/v3/role_assignments?scope.system=all', headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_role_assignments_by_domain(self):
        with self.test_client() as c:
            c.get('/v3/role_assignments?scope.domain.id=%s' % self.domain_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_role_assignments_by_other_project(self):
        project1 = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        with self.test_client() as c:
            c.get('/v3/role_assignments?scope.project.id=%s' % project1, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_role_assignments_by_other_project_user(self):
        assignments = self._setup_test_role_assignments()
        user_id = assignments['user_id']
        with self.test_client() as c:
            c.get('/v3/role_assignments?user.id=%s' % user_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_role_assignments_by_other_project_group(self):
        assignments = self._setup_test_role_assignments()
        group_id = assignments['group_id']
        with self.test_client() as c:
            c.get('/v3/role_assignments?group.id=%s' % group_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)