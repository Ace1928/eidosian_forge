import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_get_project_limit_without_role_assignment(self):
    project_limit_id, _ = _create_limits_and_dependencies()
    with self.test_client() as c:
        c.get('/v3/limits/%s' % project_limit_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)