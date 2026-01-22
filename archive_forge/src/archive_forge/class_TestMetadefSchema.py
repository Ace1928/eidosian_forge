from openstack.image.v2 import metadef_schema
from openstack.tests.unit import base
class TestMetadefSchema(base.TestCase):

    def test_basic(self):
        sot = metadef_schema.MetadefSchema()
        self.assertIsNone(sot.resource_key)
        self.assertIsNone(sot.resources_key)
        self.assertEqual('/schemas/metadefs', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)

    def test_make_it(self):
        sot = metadef_schema.MetadefSchema(**EXAMPLE)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['properties'], sot.properties)
        self.assertEqual(EXAMPLE['additionalProperties'], sot.additional_properties)
        self.assertEqual(EXAMPLE['definitions'], sot.definitions)
        self.assertEqual(EXAMPLE['required'], sot.required)