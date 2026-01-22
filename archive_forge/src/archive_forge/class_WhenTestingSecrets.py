import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
class WhenTestingSecrets(test_client.BaseEntityResource):

    def setUp(self):
        self._setUp('secrets')
        self.secret = SecretData()
        self.manager = self.client.secrets
        self.consumers_post_resource = self.entity_href + '/consumers/'
        self.consumers_delete_resource = self.entity_href + '/consumers'

    def test_should_entity_str(self):
        secret_obj = self.manager.create(name=self.secret.name)
        self.assertIn(self.secret.name, str(secret_obj))

    def test_should_entity_repr(self):
        secret_obj = self.manager.create(name=self.secret.name)
        self.assertIn('name="{0}"'.format(self.secret.name), repr(secret_obj))

    def test_should_store_via_constructor(self):
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        secret = self.manager.create(name=self.secret.name, payload=self.secret.payload)
        secret_href = secret.store()
        self.assertEqual(self.entity_href, secret_href)
        secret_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.secret.name, secret_req['name'])
        self.assertEqual(self.secret.payload, secret_req['payload'])

    def test_should_store_via_attributes(self):
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        secret = self.manager.create()
        secret.name = self.secret.name
        secret.payload = self.secret.payload
        secret_href = secret.store()
        self.assertEqual(self.entity_href, secret_href)
        secret_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.secret.name, secret_req['name'])
        self.assertEqual(self.secret.payload, secret_req['payload'])

    def test_should_store_binary_type_as_octet_stream(self):
        """We use bytes as the canonical binary type.

        The client should base64 encode the payload before sending the
        request.
        """
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        binary_payload = b'F\x130\x89f\x8e\xd9\xa1\x0e\x1f\r\xf67uu\x8b'
        secret = self.manager.create()
        secret.name = self.secret.name
        secret.payload = binary_payload
        secret.store()
        secret_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.secret.name, secret_req['name'])
        self.assertEqual('application/octet-stream', secret_req['payload_content_type'])
        self.assertEqual('base64', secret_req['payload_content_encoding'])
        self.assertNotEqual(binary_payload, secret_req['payload'])

    def test_should_store_text_type_as_text_plain(self):
        """We use unicode string as the canonical text type."""
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        text_payload = 'time for an ice cold 🍺'
        secret = self.manager.create()
        secret.payload = text_payload
        secret.store()
        secret_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(text_payload, secret_req['payload'])
        self.assertEqual('text/plain', secret_req['payload_content_type'])

    def test_should_store_with_deprecated_content_type(self):
        """DEPRECATION WARNING

        Manually setting the payload_content_type is deprecated and will be
        removed in a future release.
        """
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        payload = 'I should be octet-stream'
        payload_content_type = 'text/plain'
        secret = self.manager.create()
        secret.payload = payload
        secret.payload_content_type = payload_content_type
        secret.store()
        secret_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(payload, secret_req['payload'])
        self.assertEqual(payload_content_type, secret_req['payload_content_type'])

    def test_should_store_with_deprecated_content_encoding(self):
        """DEPRECATION WARNING

        Manually setting the payload_content_encoding is deprecated and will be
        removed in a future release.
        """
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        encoded_payload = base64.b64encode(b'F\x130\x89f\x8e\xd9\xa1\x0e\x1f\r\xf67uu\x8b').decode('UTF-8')
        payload_content_type = 'application/octet-stream'
        payload_content_encoding = 'base64'
        secret = self.manager.create()
        secret.payload = encoded_payload
        secret.payload_content_type = payload_content_type
        secret.payload_content_encoding = payload_content_encoding
        secret.store()
        secret_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(encoded_payload, secret_req['payload'])
        self.assertEqual(payload_content_type, secret_req['payload_content_type'])
        self.assertEqual(payload_content_encoding, secret_req['payload_content_encoding'])

    def test_should_be_immutable_after_submit(self):
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        secret = self.manager.create(name=self.secret.name, payload=self.secret.payload)
        secret_href = secret.store()
        self.assertEqual(self.entity_href, secret_href)
        attributes = ['name', 'expiration', 'algorithm', 'bit_length', 'mode', 'payload_content_type', 'payload_content_encoding', 'secret_type']
        for attr in attributes:
            try:
                setattr(secret, attr, 'test')
                self.fail("didn't raise an ImmutableException exception")
            except base.ImmutableException:
                pass

    def test_should_not_be_able_to_set_generated_attributes(self):
        secret = self.manager.create()
        attributes = ['secret_ref', 'created', 'updated', 'content_types', 'status']
        for attr in attributes:
            try:
                setattr(secret, attr, 'test')
                self.fail("didn't raise an AttributeError exception")
            except AttributeError:
                pass

    def test_should_get_lazy(self, secret_ref=None):
        secret_ref = secret_ref or self.entity_href
        data = self.secret.get_dict(secret_ref)
        m = self.responses.get(self.entity_href, json=data)
        secret = self.manager.get(secret_ref=secret_ref)
        self.assertIsInstance(secret, secrets.Secret)
        self.assertEqual(secret_ref, secret.secret_ref)
        self.assertFalse(m.called)
        self.assertEqual(self.secret.name, secret.name)
        self.assertEqual(self.entity_href, m.last_request.url)

    def test_should_get_lazy_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_get_lazy(bad_href)

    def test_should_get_lazy_using_only_uuid(self):
        self.test_should_get_lazy(self.entity_id)

    def test_should_get_acls_lazy(self):
        data = self.secret.get_dict(self.entity_href)
        m = self.responses.get(self.entity_href, json=data)
        acl_data = {'read': {'project-access': True, 'users': ['u1']}}
        acl_ref = self.entity_href + '/acl'
        n = self.responses.get(acl_ref, json=acl_data)
        secret = self.manager.get(secret_ref=self.entity_href)
        self.assertIsNotNone(secret)
        self.assertEqual(self.secret.name, secret.name)
        self.assertTrue(m.called)
        self.assertFalse(n.called)
        self.assertEqual(['u1'], secret.acls.read.users)
        self.assertTrue(secret.acls.read.project_access)
        self.assertIsInstance(secret.acls, acls.SecretACL)
        self.assertEqual(acl_ref, n.last_request.url)

    def test_should_get_payload_only_when_content_type_is_set(self):
        """DEPRECATION WARNING

        Manually setting the payload_content_type is deprecated and will be
        removed in a future release.
        """
        m = self.responses.get(self.entity_href, request_headers={'Accept': 'application/json'}, json=self.secret.get_dict(self.entity_href))
        n = self.responses.get(self.entity_payload_href, request_headers={'Accept': 'text/plain'}, text=self.secret.payload)
        secret = self.manager.get(secret_ref=self.entity_href, payload_content_type=self.secret.payload_content_type)
        self.assertIsInstance(secret, secrets.Secret)
        self.assertEqual(self.entity_href, secret.secret_ref)
        self.assertFalse(m.called)
        self.assertFalse(n.called)
        self.assertEqual(self.secret.payload, secret.payload)
        self.assertFalse(m.called)
        self.assertTrue(n.called)
        self.assertEqual(self.entity_payload_href, n.last_request.url)

    def test_should_fetch_metadata_to_get_payload(self):
        content_types_dict = {'default': 'text/plain'}
        data = self.secret.get_dict(self.entity_href, content_types_dict=content_types_dict)
        metadata_response = self.responses.get(self.entity_href, request_headers={'Accept': 'application/json'}, json=data)
        request_headers = {'Accept': 'text/plain'}
        decryption_response = self.responses.get(self.entity_payload_href, request_headers=request_headers, text=self.secret.payload)
        secret = self.manager.get(secret_ref=self.entity_href)
        self.assertIsInstance(secret, secrets.Secret)
        self.assertEqual(self.entity_href, secret.secret_ref)
        self.assertFalse(metadata_response.called)
        self.assertFalse(decryption_response.called)
        self.assertEqual(self.secret.payload, secret.payload)
        self.assertTrue(metadata_response.called)
        self.assertTrue(decryption_response.called)
        self.assertEqual(self.entity_href, metadata_response.last_request.url)
        self.assertEqual(self.entity_payload_href, decryption_response.last_request.url)

    def test_should_decrypt_when_content_type_is_set(self):
        """DEPRECATION WARNING

        Manually setting the payload_content_type is deprecated and will be
        removed in a future release.
        """
        decrypted = b'decrypted text here'
        request_headers = {'Accept': 'application/octet-stream'}
        m = self.responses.get(self.entity_payload_href, request_headers=request_headers, content=decrypted)
        secret = self.manager.get(secret_ref=self.entity_href, payload_content_type='application/octet-stream')
        secret_payload = secret.payload
        self.assertEqual(decrypted, secret_payload)
        self.assertEqual(self.entity_payload_href, m.last_request.url)

    def test_should_decrypt(self, secret_ref=None):
        secret_ref = secret_ref or self.entity_href
        content_types_dict = {'default': 'application/octet-stream'}
        json = self.secret.get_dict(secret_ref, content_types_dict)
        metadata_response = self.responses.get(self.entity_href, request_headers={'Accept': 'application/json'}, json=json)
        decrypted = b'decrypted text here'
        request_headers = {'Accept': 'application/octet-stream'}
        decryption_response = self.responses.get(self.entity_payload_href, request_headers=request_headers, content=decrypted)
        secret = self.manager.get(secret_ref=secret_ref)
        secret_payload = secret.payload
        self.assertEqual(decrypted, secret_payload)
        self.assertEqual(self.entity_href, metadata_response.last_request.url)
        self.assertEqual(self.entity_payload_href, decryption_response.last_request.url)

    def test_should_decrypt_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_decrypt(bad_href)

    def _mock_delete_secret(self):
        self.responses.delete(self.entity_href, status_code=204)

    def _delete_from_manager(self, secret_ref, force=False):
        mock_get_secret_for_client(self.client)
        self._mock_delete_secret()
        self.manager.delete(secret_ref=secret_ref, force=force)
        self.assertEqual(self.entity_href, self.responses.last_request.url)

    def test_should_delete_from_manager(self):
        self._delete_from_manager(self.entity_href)

    def test_should_delete_from_manager_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self._delete_from_manager(secret_ref=bad_href)

    def test_should_delete_from_manager_using_only_uuid(self):
        self._delete_from_manager(secret_ref=self.entity_id)

    def test_should_delete_from_object(self, secref_ref=None):
        secref_ref = secref_ref or self.entity_href
        data = {'secret_ref': secref_ref}
        self.responses.post(self.entity_base + '/', json=data)
        secret = self.manager.create()
        secret.payload = None
        secret.store()
        self.assertEqual(secref_ref, secret.secret_ref)
        self.responses.delete(self.entity_href, status_code=204)
        secret.delete()
        self.assertEqual(self.entity_href, self.responses.last_request.url)

    def test_should_delete_from_object_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_delete_from_object(bad_href)

    def test_should_delete_from_object_using_only_uuid(self):
        self.test_should_delete_from_object(self.entity_id)

    def test_should_update_from_manager(self, secret_ref=None):
        text_payload = 'time for an ice cold 🍺'
        secret_ref = secret_ref or self.entity_href
        self.responses.put(self.entity_href, status_code=204)
        self.manager.update(secret_ref=secret_ref, payload=text_payload)
        self.assertEqual(self.entity_href, self.responses.last_request.url)

    def test_should_update_from_manager_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_update_from_manager(bad_href)

    def test_should_update_from_manager_using_only_uuid(self):
        self.test_should_update_from_manager(self.entity_id)

    def test_should_update_from_object(self, secref_ref=None):
        secref_ref = secref_ref or self.entity_href
        data = {'secret_ref': secref_ref}
        self.responses.post(self.entity_base + '/', json=data)
        secret = self.manager.create()
        secret.payload = None
        secret.store()
        self.assertEqual(secref_ref, secret.secret_ref)
        text_payload = 'time for an ice cold 🍺'
        self.responses.put(self.entity_href, status_code=204)
        secret.payload = text_payload
        secret.update()
        self.assertEqual(self.entity_href, self.responses.last_request.url)
        self.assertEqual(text_payload, secret.payload)

    def test_should_update_from_object_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_update_from_object(bad_href)

    def test_should_update_from_object_using_only_uuid(self):
        self.test_should_update_from_object(self.entity_id)

    def test_should_get_list(self):
        secret_resp = self.secret.get_dict(self.entity_href)
        data = {'secrets': [secret_resp for v in range(3)]}
        m = self.responses.get(self.entity_base, json=data)
        secrets_list = self.manager.list(limit=10, offset=5)
        self.assertTrue(len(secrets_list) == 3)
        self.assertIsInstance(secrets_list[0], secrets.Secret)
        self.assertEqual(self.entity_href, secrets_list[0].secret_ref)
        self.assertEqual(self.entity_base, m.last_request.url.split('?')[0])
        self.assertEqual(['10'], m.last_request.qs['limit'])
        self.assertEqual(['5'], m.last_request.qs['offset'])

    def test_should_fail_get_invalid_secret(self):
        self.assertRaises(ValueError, self.manager.get, **{'secret_ref': '12345'})

    def test_should_fail_update_zero(self):
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        secret = self.manager.create()
        secret.payload = None
        secret.store()
        self.responses.put(self.entity_href, status_code=204)
        secret.payload = 0
        self.assertRaises(exceptions.PayloadException, secret.update)

    def test_should_fail_store_zero(self):
        data = {'secret_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        secret = self.manager.create()
        secret.name = self.secret.name
        secret.payload = 0
        self.assertRaises(exceptions.PayloadException, secret.store)

    def test_should_fail_decrypt_no_content_types(self):
        data = self.secret.get_dict(self.entity_href)
        self.responses.get(self.entity_href, json=data)
        secret = self.manager.get(secret_ref=self.entity_href)
        self.assertIsNone(secret.payload)

    def test_should_fail_decrypt_no_default_content_type(self):
        content_types_dict = {'no-default': 'application/octet-stream'}
        data = self.secret.get_dict(self.entity_href, content_types_dict)
        self.responses.get(self.entity_href, json=data)
        secret = self.manager.get(secret_ref=self.entity_href)
        self.assertIsNone(secret.payload)

    def test_should_fail_delete_no_href(self):
        self.assertRaises(ValueError, self.manager.delete, None)

    def test_should_get_total(self):
        self.responses.get(self.entity_base, json={'total': 1})
        total = self.manager.total()
        self.assertEqual(1, total)

    def test_get_formatted_data(self):
        data = self.secret.get_dict(self.entity_href)
        self.responses.get(self.entity_href, json=data)
        secret = self.manager.get(secret_ref=self.entity_href)
        f_data = secret._get_formatted_data()
        self.assertEqual(timeutils.parse_isotime(data['created']).isoformat(), f_data[2])