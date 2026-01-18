import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_create_trust(self):
    trust_data = self.trust_data['trust']
    trust_data['trustor_user_id'] = self.user_id
    json = {'trust': trust_data}
    json['trust']['roles'] = self.trust_data['roles']
    with self.test_client() as c:
        c.post('/v3/OS-TRUST/trusts', json=json, headers=self.headers, expected_status_code=http.client.FORBIDDEN)