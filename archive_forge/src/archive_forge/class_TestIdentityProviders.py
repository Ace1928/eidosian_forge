import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
class TestIdentityProviders(base.V3ClientTestCase):

    def test_idp_create(self):
        idp_id = uuid.uuid4().hex
        idp = self.client.federation.identity_providers.create(id=idp_id)
        self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
        self.assertEqual(idp_id, idp.id)
        self.assertEqual([], idp.remote_ids)
        self.assertFalse(idp.enabled)

    def test_idp_create_enabled_true(self):
        idp_id = uuid.uuid4().hex
        idp = self.client.federation.identity_providers.create(id=idp_id, enabled=True)
        self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
        self.assertEqual(idp_id, idp.id)
        self.assertEqual([], idp.remote_ids)
        self.assertTrue(idp.enabled)

    def test_idp_create_with_remote_ids(self):
        idp_id = uuid.uuid4().hex
        remote_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
        idp = self.client.federation.identity_providers.create(id=idp_id, enabled=True, remote_ids=remote_ids)
        self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
        self.assertEqual(idp_id, idp.id)
        self.assertIn(remote_ids[0], idp.remote_ids)
        self.assertIn(remote_ids[1], idp.remote_ids)
        self.assertTrue(idp.enabled)

    def test_idp_list(self):
        idp_ids = []
        for _ in range(3):
            idp_id = uuid.uuid4().hex
            self.client.federation.identity_providers.create(id=idp_id)
            self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
            idp_ids.append(idp_id)
        idp_list = self.client.federation.identity_providers.list()
        fetched_ids = [fetched_idp.id for fetched_idp in idp_list]
        for idp_id in idp_ids:
            self.assertIn(idp_id, fetched_ids)

    def test_idp_get(self):
        idp_id = uuid.uuid4().hex
        remote_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
        idp_create = self.client.federation.identity_providers.create(id=idp_id, enabled=True, remote_ids=remote_ids)
        self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
        idp_get = self.client.federation.identity_providers.get(idp_id)
        self.assertEqual(idp_create.id, idp_get.id)
        self.assertEqual(idp_create.enabled, idp_get.enabled)
        self.assertIn(idp_create.remote_ids[0], idp_get.remote_ids)
        self.assertIn(idp_create.remote_ids[1], idp_get.remote_ids)

    def test_idp_delete(self):
        idp_id = uuid.uuid4().hex
        self.client.federation.identity_providers.create(id=idp_id)
        self.client.federation.identity_providers.get(idp_id)
        self.client.federation.identity_providers.delete(idp_id)
        self.assertRaises(http.NotFound, self.client.federation.identity_providers.get, idp_id)
        idp_list = self.client.federation.identity_providers.list()
        fetched_ids = [fetched_idp.id for fetched_idp in idp_list]
        self.assertNotIn(idp_id, fetched_ids)