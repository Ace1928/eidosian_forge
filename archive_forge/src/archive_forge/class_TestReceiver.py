from openstack.clustering.v1 import receiver
from openstack.tests.unit import base
class TestReceiver(base.TestCase):

    def setUp(self):
        super(TestReceiver, self).setUp()

    def test_basic(self):
        sot = receiver.Receiver()
        self.assertEqual('receiver', sot.resource_key)
        self.assertEqual('receivers', sot.resources_key)
        self.assertEqual('/receivers', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_instantiate(self):
        sot = receiver.Receiver(**FAKE)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['type'], sot.type)
        self.assertEqual(FAKE['cluster_id'], sot.cluster_id)
        self.assertEqual(FAKE['action'], sot.action)
        self.assertEqual(FAKE['params'], sot.params)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)
        self.assertEqual(FAKE['user'], sot.user_id)
        self.assertEqual(FAKE['project'], sot.project_id)
        self.assertEqual(FAKE['domain'], sot.domain_id)
        self.assertEqual(FAKE['channel'], sot.channel)