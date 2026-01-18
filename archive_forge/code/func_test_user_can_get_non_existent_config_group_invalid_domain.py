import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_non_existent_config_group_invalid_domain(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    config = {'ldap': {'url': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(domain['id'], config)
    invalid_domain_id = uuid.uuid4().hex
    with self.test_client() as c:
        c.get('/v3/domains/%s/config/ldap' % invalid_domain_id, headers=self.headers, expected_status_code=http.client.NOT_FOUND)