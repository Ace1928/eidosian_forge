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
def test_user_can_filter_role_assignments_by_system_and_role(self):
    assignments = self._setup_test_role_assignments()
    self.expected = [ra for ra in self.expected if ra['role_id'] == assignments['role_id']]
    self.expected.append({'user_id': assignments['user_id'], 'system': 'all', 'role_id': assignments['role_id']})
    self.expected.append({'group_id': assignments['group_id'], 'system': 'all', 'role_id': assignments['role_id']})
    role_id = assignments['role_id']
    with self.test_client() as c:
        r = c.get('/v3/role_assignments?scope.system=all&role.id=%s' % role_id, headers=self.headers)
        self.assertEqual(len(self.expected), len(r.json['role_assignments']))
        actual = self._extract_role_assignments_from_response_body(r)
        for assignment in actual:
            self.assertIn(assignment, self.expected)