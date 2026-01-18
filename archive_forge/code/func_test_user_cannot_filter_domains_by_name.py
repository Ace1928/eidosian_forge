import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import domain as dp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_filter_domains_by_name(self):
    domain_name = uuid.uuid4().hex
    domain = unit.new_domain_ref(name=domain_name)
    domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
    PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    with self.test_client() as c:
        c.get('/v3/domains?name=%s' % domain_name, headers=self.headers, expected_status_code=http.client.FORBIDDEN)