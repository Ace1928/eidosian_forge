import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
class StrictTwoLevelLimitsTestCase(LimitsTestCase):

    def setUp(self):
        super(StrictTwoLevelLimitsTestCase, self).setUp()
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)
        self.system_admin_token = self.get_system_scoped_token()
        domain_ref = {'domain': {'name': 'A', 'enabled': True}}
        response = self.post('/domains', body=domain_ref)
        self.domain_A = response.json_body['domain']
        project_ref = {'project': {'name': 'B', 'enabled': True, 'domain_id': self.domain_A['id']}}
        response = self.post('/projects', body=project_ref)
        self.project_B = response.json_body['project']
        project_ref = {'project': {'name': 'C', 'enabled': True, 'domain_id': self.domain_A['id']}}
        response = self.post('/projects', body=project_ref)
        self.project_C = response.json_body['project']
        domain_ref = {'domain': {'name': 'D', 'enabled': True}}
        response = self.post('/domains', body=domain_ref)
        self.domain_D = response.json_body['domain']
        project_ref = {'project': {'name': 'E', 'enabled': True, 'domain_id': self.domain_D['id']}}
        response = self.post('/projects', body=project_ref)
        self.project_E = response.json_body['project']
        project_ref = {'project': {'name': 'F', 'enabled': True, 'domain_id': self.domain_D['id']}}
        response = self.post('/projects', body=project_ref)
        self.project_F = response.json_body['project']

    def config_overrides(self):
        super(StrictTwoLevelLimitsTestCase, self).config_overrides()
        self.config_fixture.config(group='unified_limit', enforcement_model='strict_two_level')

    def test_create_child_limit(self):
        ref = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=20)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=15)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=18)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)

    def test_create_child_limit_break_hierarchical_tree(self):
        ref = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=20)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=15)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=21)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_create_child_with_default_parent(self):
        ref = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=11)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_create_parent_limit(self):
        ref = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=12)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)

    def test_create_parent_limit_break_hierarchical_tree(self):
        ref = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=8)
        self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_create_multi_limits(self):
        ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=12)
        ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
        ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
        ref_D = unit.new_limit_ref(domain_id=self.domain_D['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
        ref_E = unit.new_limit_ref(project_id=self.project_E['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
        ref_F = unit.new_limit_ref(project_id=self.project_F['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=4)
        self.post('/limits', body={'limits': [ref_A, ref_B, ref_C, ref_D, ref_E, ref_F]}, token=self.system_admin_token, expected_status=http.client.CREATED)

    def test_create_multi_limits_invalid_input(self):
        ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=12)
        ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
        ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
        ref_D = unit.new_limit_ref(domain_id=self.domain_D['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
        ref_E = unit.new_limit_ref(project_id=self.project_E['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
        ref_F = unit.new_limit_ref(project_id=self.project_F['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
        self.post('/limits', body={'limits': [ref_A, ref_B, ref_C, ref_D, ref_E, ref_F]}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_create_multi_limits_break_hierarchical_tree(self):
        ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=12)
        ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
        ref_E = unit.new_limit_ref(project_id=self.project_E['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
        self.post('/limits', body={'limits': [ref_A, ref_B, ref_E]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
        ref_D = unit.new_limit_ref(domain_id=self.domain_D['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=4)
        self.post('/limits', body={'limits': [ref_C, ref_D]}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_update_child_limit(self):
        ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
        ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=6)
        ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=7)
        self.post('/limits', body={'limits': [ref_A, ref_B]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        r = self.post('/limits', body={'limits': [ref_C]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_dict = {'resource_limit': 9}
        self.patch('/limits/%s' % r.result['limits'][0]['id'], body={'limit': update_dict}, token=self.system_admin_token, expected_status=http.client.OK)

    def test_update_child_limit_break_hierarchical_tree(self):
        ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
        ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=6)
        ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=7)
        self.post('/limits', body={'limits': [ref_A, ref_B]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        r = self.post('/limits', body={'limits': [ref_C]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_dict = {'resource_limit': 11}
        self.patch('/limits/%s' % r.result['limits'][0]['id'], body={'limit': update_dict}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_update_child_limit_with_default_parent(self):
        ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=7)
        r = self.post('/limits', body={'limits': [ref_C]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_dict = {'resource_limit': 9}
        self.patch('/limits/%s' % r.result['limits'][0]['id'], body={'limit': update_dict}, token=self.system_admin_token, expected_status=http.client.OK)
        update_dict = {'resource_limit': 11}
        self.patch('/limits/%s' % r.result['limits'][0]['id'], body={'limit': update_dict}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)

    def test_update_parent_limit(self):
        ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
        ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=6)
        ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=7)
        r = self.post('/limits', body={'limits': [ref_A]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        self.post('/limits', body={'limits': [ref_B, ref_C]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_dict = {'resource_limit': 8}
        self.patch('/limits/%s' % r.result['limits'][0]['id'], body={'limit': update_dict}, token=self.system_admin_token, expected_status=http.client.OK)

    def test_update_parent_limit_break_hierarchical_tree(self):
        ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
        ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=6)
        ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=7)
        r = self.post('/limits', body={'limits': [ref_A]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        self.post('/limits', body={'limits': [ref_B, ref_C]}, token=self.system_admin_token, expected_status=http.client.CREATED)
        update_dict = {'resource_limit': 6}
        self.patch('/limits/%s' % r.result['limits'][0]['id'], body={'limit': update_dict}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)