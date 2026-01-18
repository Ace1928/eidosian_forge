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
def test_user_cannot_list_assignments_for_subtree_in_other_domain(self):
    assignments = self._setup_test_role_assignments()
    with self.test_client() as c:
        c.get('/v3/role_assignments?scope.project.id=%s&include_subtree' % assignments['project_id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)