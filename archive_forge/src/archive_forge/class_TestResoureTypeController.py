import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
class TestResoureTypeController(testtools.TestCase):

    def setUp(self):
        super(TestResoureTypeController, self).setUp()
        self.api = utils.FakeAPI(data_fixtures)
        self.schema_api = utils.FakeSchemaAPI(schema_fixtures)
        self.controller = base.BaseResourceTypeController(self.api, self.schema_api, metadefs.ResourceTypeController)

    def test_list_resource_types(self):
        resource_types = self.controller.list()
        names = [rt.name for rt in resource_types]
        self.assertEqual([RESOURCE_TYPE1, RESOURCE_TYPE2], names)

    def test_get_resource_types(self):
        resource_types = self.controller.get(NAMESPACE1)
        self.assertEqual([RESOURCE_TYPE3, RESOURCE_TYPE4], resource_types)

    def test_associate_resource_types(self):
        resource_types = self.controller.associate(NAMESPACE1, name=RESOURCE_TYPENEW)
        self.assertEqual(RESOURCE_TYPENEW, resource_types['name'])

    def test_associate_resource_types_invalid_property(self):
        longer = '1234' * 50
        properties = {'name': RESOURCE_TYPENEW, 'prefix': longer}
        self.assertRaises(TypeError, self.controller.associate, NAMESPACE1, **properties)

    def test_deassociate_resource_types(self):
        self.controller.deassociate(NAMESPACE1, RESOURCE_TYPE1)
        expect = [('DELETE', '/v2/metadefs/namespaces/%s/resource_types/%s' % (NAMESPACE1, RESOURCE_TYPE1), {}, None)]
        self.assertEqual(expect, self.api.calls)