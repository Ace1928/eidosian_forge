import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
class TestPropertyController(testtools.TestCase):

    def setUp(self):
        super(TestPropertyController, self).setUp()
        self.api = utils.FakeAPI(data_fixtures)
        self.schema_api = utils.FakeSchemaAPI(schema_fixtures)
        self.controller = base.BaseController(self.api, self.schema_api, metadefs.PropertyController)

    def test_list_property(self):
        properties = self.controller.list(NAMESPACE1)
        actual = [prop.name for prop in properties]
        self.assertEqual(sorted([PROPERTY1, PROPERTY2]), sorted(actual))

    def test_get_property(self):
        prop = self.controller.get(NAMESPACE1, PROPERTY1)
        self.assertEqual(PROPERTY1, prop.name)

    def test_create_property(self):
        properties = {'name': PROPERTYNEW, 'title': 'TITLE', 'type': 'string'}
        obj = self.controller.create(NAMESPACE1, **properties)
        self.assertEqual(PROPERTYNEW, obj.name)

    def test_create_property_invalid_property(self):
        properties = {'namespace': NAMESPACE1}
        self.assertRaises(TypeError, self.controller.create, **properties)

    def test_update_property(self):
        properties = {'description': 'UPDATED_DESCRIPTION'}
        prop = self.controller.update(NAMESPACE1, PROPERTY1, **properties)
        self.assertEqual(PROPERTY1, prop.name)

    def test_update_property_invalid_property(self):
        properties = {'type': 'INVALID'}
        self.assertRaises(TypeError, self.controller.update, NAMESPACE1, PROPERTY1, **properties)

    def test_update_property_disallowed_fields(self):
        properties = {'description': 'UPDATED_DESCRIPTION'}
        self.controller.update(NAMESPACE1, PROPERTY1, **properties)
        actual = self.api.calls
        _disallowed_fields = ['created_at', 'updated_at']
        for key in actual[1][3]:
            self.assertNotIn(key, _disallowed_fields)

    def test_delete_property(self):
        self.controller.delete(NAMESPACE1, PROPERTY1)
        expect = [('DELETE', '/v2/metadefs/namespaces/%s/properties/%s' % (NAMESPACE1, PROPERTY1), {}, None)]
        self.assertEqual(expect, self.api.calls)

    def test_delete_all_properties(self):
        self.controller.delete_all(NAMESPACE1)
        expect = [('DELETE', '/v2/metadefs/namespaces/%s/properties' % NAMESPACE1, {}, None)]
        self.assertEqual(expect, self.api.calls)