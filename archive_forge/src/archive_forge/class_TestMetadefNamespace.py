from unittest import mock
from keystoneauth1 import adapter
from openstack import exceptions
from openstack.image.v2 import metadef_namespace
from openstack.tests.unit import base
class TestMetadefNamespace(base.TestCase):

    def test_basic(self):
        sot = metadef_namespace.MetadefNamespace()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('namespaces', sot.resources_key)
        self.assertEqual('/metadefs/namespaces', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_delete)

    def test_make_it(self):
        sot = metadef_namespace.MetadefNamespace(**EXAMPLE)
        self.assertEqual(EXAMPLE['namespace'], sot.namespace)
        self.assertEqual(EXAMPLE['visibility'], sot.visibility)
        self.assertEqual(EXAMPLE['owner'], sot.owner)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['protected'], sot.is_protected)
        self.assertEqual(EXAMPLE['display_name'], sot.display_name)
        self.assertEqual(EXAMPLE['resource_type_associations'], sot.resource_type_associations)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'resource_types': 'resource_types', 'sort_dir': 'sort_dir', 'sort_key': 'sort_key', 'visibility': 'visibility'}, sot._query_mapping._mapping)

    @mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
    def test_delete_all_properties(self):
        sot = metadef_namespace.MetadefNamespace(**EXAMPLE)
        session = mock.Mock(spec=adapter.Adapter)
        sot._translate_response = mock.Mock()
        sot.delete_all_properties(session)
        session.delete.assert_called_with('metadefs/namespaces/OS::Cinder::Volumetype/properties')