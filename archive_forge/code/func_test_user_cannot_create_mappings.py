import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_create_mappings(self):
    create = {'mapping': {'id': uuid.uuid4().hex, 'rules': [{'local': [{'user': {'name': '{0}'}}], 'remote': [{'type': 'UserName'}]}]}}
    mapping_id = create['mapping']['id']
    with self.test_client() as c:
        c.put('/v3/OS-FEDERATION/mappings/%s' % mapping_id, json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)