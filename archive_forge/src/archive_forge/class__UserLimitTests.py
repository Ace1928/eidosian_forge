import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _UserLimitTests(object):
    """Common default functionality for all users except system admins."""

    def test_user_can_get_limit_model(self):
        with self.test_client() as c:
            c.get('/v3/limits/model', headers=self.headers)

    def test_user_can_get_a_limit(self):
        limit_id, _ = _create_limits_and_dependencies()
        with self.test_client() as c:
            r = c.get('/v3/limits/%s' % limit_id, headers=self.headers)
            self.assertEqual(limit_id, r.json['limit']['id'])

    def test_user_can_list_limits(self):
        project_limit_id, domain_limit_id = _create_limits_and_dependencies()
        with self.test_client() as c:
            r = c.get('/v3/limits', headers=self.headers)
            self.assertTrue(len(r.json['limits']) == 2)
            result = []
            for limit in r.json['limits']:
                result.append(limit['id'])
            self.assertIn(project_limit_id, result)
            self.assertIn(domain_limit_id, result)

    def test_user_cannot_create_limits(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        registered_limit = unit.new_registered_limit_ref(service_id=service['id'], id=uuid.uuid4().hex)
        registered_limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
        registered_limit = registered_limits[0]
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        create = {'limits': [unit.new_limit_ref(project_id=project['id'], service_id=service['id'], resource_name=registered_limit['resource_name'], resource_limit=5)]}
        with self.test_client() as c:
            c.post('/v3/limits', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_limits(self):
        limit_id, _ = _create_limits_and_dependencies()
        update = {'limits': {'description': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/limits/%s' % limit_id, json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_limits(self):
        limit_id, _ = _create_limits_and_dependencies()
        with self.test_client() as c:
            c.delete('/v3/limits/%s' % limit_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)