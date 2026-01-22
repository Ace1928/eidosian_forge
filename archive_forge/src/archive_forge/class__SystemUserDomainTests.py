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
class _SystemUserDomainTests(object):

    def test_user_can_list_domains(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            r = c.get('/v3/domains', headers=self.headers)
            domain_ids = []
            for domain in r.json['domains']:
                domain_ids.append(domain['id'])
            self.assertIn(domain['id'], domain_ids)

    def test_user_can_filter_domains_by_name(self):
        domain_name = uuid.uuid4().hex
        domain = unit.new_domain_ref(name=domain_name)
        domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
        PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            r = c.get('/v3/domains?name=%s' % domain_name, headers=self.headers)
            self.assertEqual(1, len(r.json['domains']))
            self.assertEqual(domain['id'], r.json['domains'][0]['id'])

    def test_user_can_filter_domains_by_enabled(self):
        enabled_domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        disabled_domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref(enabled=False))
        with self.test_client() as c:
            r = c.get('/v3/domains?enabled=true', headers=self.headers)
            enabled_domain_ids = []
            for domain in r.json['domains']:
                enabled_domain_ids.append(domain['id'])
            self.assertIn(enabled_domain['id'], enabled_domain_ids)
            self.assertNotIn(disabled_domain['id'], enabled_domain_ids)
            r = c.get('/v3/domains?enabled=false', headers=self.headers)
            disabled_domain_ids = []
            for domain in r.json['domains']:
                disabled_domain_ids.append(domain['id'])
            self.assertIn(disabled_domain['id'], disabled_domain_ids)
            self.assertNotIn(enabled_domain['id'], disabled_domain_ids)

    def test_user_can_get_a_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            r = c.get('/v3/domains/%s' % domain['id'], headers=self.headers)
            self.assertEqual(domain['id'], r.json['domain']['id'])