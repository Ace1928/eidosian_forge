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
def test_user_can_filter_domains_by_name(self):
    second_domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    with self.test_client() as c:
        r = c.get('/v3/domains?name=%s' % self.domain['name'], headers=self.headers)
        self.assertEqual(1, len(r.json['domains']))
        self.assertNotIn(second_domain['id'], [d['id'] for d in r.json['domains']])
        self.assertEqual(self.domain['id'], r.json['domains'][0]['id'])
        r = c.get('/v3/domains?name=%s' % second_domain['name'], headers=self.headers)
        self.assertEqual(0, len(r.json['domains']))