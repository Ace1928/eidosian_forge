import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _UserCredentialTests(object):
    """Test cases for anyone that has a valid user token."""

    def test_user_can_create_credentials_for_themselves(self):
        create = {'credential': {'blob': uuid.uuid4().hex, 'user_id': self.user_id, 'type': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.post('/v3/credentials', json=create, headers=self.headers)

    def test_user_can_get_their_credentials(self):
        with self.test_client() as c:
            create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': self.user_id}}
            r = c.post('/v3/credentials', json=create, headers=self.headers)
            credential_id = r.json['credential']['id']
            path = '/v3/credentials/%s' % credential_id
            r = c.get(path, headers=self.headers)
            self.assertEqual(self.user_id, r.json['credential']['user_id'])

    def test_user_can_list_their_credentials(self):
        with self.test_client() as c:
            expected = []
            for _ in range(2):
                create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': self.user_id}}
                r = c.post('/v3/credentials', json=create, headers=self.headers)
                expected.append(r.json['credential'])
            r = c.get('/v3/credentials', headers=self.headers)
            for credential in expected:
                self.assertIn(credential, r.json['credentials'])

    def test_user_can_filter_their_credentials_by_type_and_user(self):
        with self.test_client() as c:
            credential_type = uuid.uuid4().hex
            create = {'credential': {'blob': uuid.uuid4().hex, 'type': credential_type, 'user_id': self.user_id}}
            r = c.post('/v3/credentials', json=create, headers=self.headers)
            expected_credential_id = r.json['credential']['id']
            create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': self.user_id}}
            r = c.post('/v3/credentials', json=create, headers=self.headers)
            path = '/v3/credentials?type=%s' % credential_type
            r = c.get(path, headers=self.headers)
            self.assertEqual(expected_credential_id, r.json['credentials'][0]['id'])
            path = '/v3/credentials?user=%s' % self.user_id
            r = c.get(path, headers=self.headers)
            self.assertEqual(expected_credential_id, r.json['credentials'][0]['id'])

    def test_user_can_update_their_credential(self):
        with self.test_client() as c:
            create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': self.user_id}}
            r = c.post('/v3/credentials', json=create, headers=self.headers)
            credential_id = r.json['credential']['id']
            updated_blob = uuid.uuid4().hex
            update = {'credential': {'blob': updated_blob}}
            path = '/v3/credentials/%s' % credential_id
            r = c.patch(path, json=update, headers=self.headers)
            self.assertEqual(updated_blob, r.json['credential']['blob'])

    def test_user_can_delete_their_credentials(self):
        with self.test_client() as c:
            create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': self.user_id}}
            r = c.post('/v3/credentials', json=create, headers=self.headers)
            credential_id = r.json['credential']['id']
            path = '/v3/credentials/%s' % credential_id
            c.delete(path, headers=self.headers)