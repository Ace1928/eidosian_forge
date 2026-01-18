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
def test_user_cannot_filter_role_assignments_by_other_domain_project(self):
    assignments = self._setup_test_role_assignments()
    self._setup_test_role_assignments_for_domain()
    project_id = assignments['project_id']
    with self.test_client() as c:
        r = c.get('/v3/role_assignments?scope.project.id=%s' % project_id, headers=self.headers)
        self.assertEqual(0, len(r.json['role_assignments']))