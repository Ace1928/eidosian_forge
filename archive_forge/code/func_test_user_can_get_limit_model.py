import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_limit_model(self):
    with self.test_client() as c:
        c.get('/v3/limits/model', headers=self.headers)