import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_create_credentials_for_other_users(self):
    user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
    with self.test_client() as c:
        create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': user['id']}}
        c.post('/v3/credentials', json=create, headers=self.headers)