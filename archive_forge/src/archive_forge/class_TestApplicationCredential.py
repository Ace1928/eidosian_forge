from openstack.identity.v3 import application_credential
from openstack.tests.unit import base
class TestApplicationCredential(base.TestCase):

    def test_basic(self):
        sot = application_credential.ApplicationCredential()
        self.assertEqual('application_credential', sot.resource_key)
        self.assertEqual('application_credentials', sot.resources_key)
        self.assertEqual('/users/%(user_id)s/application_credentials', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = application_credential.ApplicationCredential(**EXAMPLE)
        self.assertEqual(EXAMPLE['user'], sot.user)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['secret'], sot.secret)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['expires_at'], sot.expires_at)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['roles'], sot.roles)
        self.assertEqual(EXAMPLE['links'], sot.links)
        self.assertEqual(EXAMPLE['access_rules'], sot.access_rules)