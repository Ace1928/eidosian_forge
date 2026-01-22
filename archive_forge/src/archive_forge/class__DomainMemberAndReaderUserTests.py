import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import user as up
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _DomainMemberAndReaderUserTests(object):
    """Functionality for all domain members and domain readers."""

    def test_user_cannot_create_users_within_domain(self):
        create = {'user': {'domain_id': self.domain_id, 'name': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.post('/v3/users', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_create_users_in_other_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        create = {'user': {'domain_id': domain['id'], 'name': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.post('/v3/users', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_users_within_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        update = {'user': {'email': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/users/%s' % user['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_users_in_other_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        update = {'user': {'email': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/users/%s' % user['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_non_existent_user_forbidden(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        update = {'user': {'email': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/users/%s' % user['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_users_within_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        with self.test_client() as c:
            c.delete('/v3/users/%s' % user['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_users_in_other_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        with self.test_client() as c:
            c.delete('/v3/users/%s' % user['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_non_existent_user_forbidden(self):
        with self.test_client() as c:
            c.delete('/v3/users/%s' % uuid.uuid4().hex, headers=self.headers, expected_status_code=http.client.FORBIDDEN)