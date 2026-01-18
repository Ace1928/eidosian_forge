import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_create_roles(self):
    create = {'role': unit.new_role_ref()}
    with self.test_client() as c:
        c.post('/v3/roles', json=create, headers=self.headers)