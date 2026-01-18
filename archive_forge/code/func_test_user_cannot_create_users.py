import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import user as up
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_create_users(self):
    create = {'user': {'name': uuid.uuid4().hex, 'domain': CONF.identity.default_domain_id}}
    with self.test_client() as c:
        c.post('/v3/users', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)