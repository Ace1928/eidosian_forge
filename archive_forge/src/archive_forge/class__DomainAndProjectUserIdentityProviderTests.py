import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _DomainAndProjectUserIdentityProviderTests(object):
    """Common functionality for all domain and project users."""

    def test_user_cannot_create_identity_providers(self):
        create = {'identity_provider': {'remote_ids': [uuid.uuid4().hex]}}
        with self.test_client() as c:
            c.put('/v3/OS-FEDERATION/identity_providers/%s' % uuid.uuid4().hex, json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_identity_providers(self):
        idp = PROVIDERS.federation_api.create_idp(uuid.uuid4().hex, unit.new_identity_provider_ref())
        update = {'identity_provider': {'enabled': False}}
        with self.test_client() as c:
            c.patch('/v3/OS-FEDERATION/identity_providers/%s' % idp['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_list_identity_providers(self):
        PROVIDERS.federation_api.create_idp(uuid.uuid4().hex, unit.new_identity_provider_ref())
        with self.test_client() as c:
            c.get('/v3/OS-FEDERATION/identity_providers', headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_get_an_identity_provider(self):
        idp = PROVIDERS.federation_api.create_idp(uuid.uuid4().hex, unit.new_identity_provider_ref())
        with self.test_client() as c:
            c.get('/v3/OS-FEDERATION/identity_providers/%s' % idp['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_identity_providers(self):
        idp = PROVIDERS.federation_api.create_idp(uuid.uuid4().hex, unit.new_identity_provider_ref())
        with self.test_client() as c:
            c.delete('/v3/OS-FEDERATION/identity_providers/%s' % idp['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)