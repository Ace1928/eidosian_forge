import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_update_limits(self):
    limit_id, _ = _create_limits_and_dependencies()
    update = {'limits': {'description': uuid.uuid4().hex}}
    with self.test_client() as c:
        c.patch('/v3/limits/%s' % limit_id, json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)