import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_update_domain_roles(self):
    role = PROVIDERS.role_api.create_role(uuid.uuid4().hex, unit.new_role_ref(domain_id=CONF.identity.default_domain_id))
    update = {'role': {'description': uuid.uuid4().hex}}
    with self.test_client() as c:
        c.patch('/v3/roles/%s' % role['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)