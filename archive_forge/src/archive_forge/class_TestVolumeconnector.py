from openstack.baremetal.v1 import volume_connector
from openstack.tests.unit import base
class TestVolumeconnector(base.TestCase):

    def test_basic(self):
        sot = volume_connector.VolumeConnector()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('connectors', sot.resources_key)
        self.assertEqual('/volume/connectors', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)

    def test_instantiate(self):
        sot = volume_connector.VolumeConnector(**FAKE)
        self.assertEqual(FAKE['connector_id'], sot.connector_id)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['extra'], sot.extra)
        self.assertEqual(FAKE['links'], sot.links)
        self.assertEqual(FAKE['node_uuid'], sot.node_id)
        self.assertEqual(FAKE['type'], sot.type)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)
        self.assertEqual(FAKE['uuid'], sot.id)