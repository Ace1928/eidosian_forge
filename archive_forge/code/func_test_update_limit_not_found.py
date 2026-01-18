import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_limit_not_found(self):
    update_ref = {'resource_limit': 5}
    self.patch('/limits/%s' % uuid.uuid4().hex, body={'limit': update_ref}, token=self.system_admin_token, expected_status=http.client.NOT_FOUND)