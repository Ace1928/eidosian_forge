from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
class TestAPIExtensionDescriptor(base.BaseTestCase):
    NAME = 'Test API'
    ALIAS = 'test-api'
    DESCRIPTION = 'A test API definition'
    UPDATED_TIMESTAMP = '2017-02-01T10:00:00-00:00'
    RESOURCE_ATTRIBUTE_MAP = {'ports': {}}
    SUB_RESOURCE_ATTRIBUTE_MAP = {'ports': {'debug': {}}}
    REQUIRED_EXTENSIONS = ['l3']
    OPTIONAL_EXTENSIONS = ['fw']

    def setUp(self):
        super(TestAPIExtensionDescriptor, self).setUp()
        self.extn = _APIDefinition()
        self.empty_extn = _EmptyAPIDefinition()
        self.useFixture(fixture.APIDefinitionFixture(self))

    def test__assert_api_definition_no_defn(self):
        self.assertRaises(NotImplementedError, _NoAPIDefinition._assert_api_definition)

    def test__assert_api_definition_no_attr(self):
        self.assertRaises(NotImplementedError, self.extn._assert_api_definition, attr='NOPE')

    def test_get_name(self):
        self.assertEqual(self.NAME, self.extn.get_name())

    def test_get_name_unset(self):
        self.assertRaises(NotImplementedError, _EmptyAPIDefinition.get_name)

    def test_get_alias(self):
        self.assertEqual(self.ALIAS, self.extn.get_alias())

    def test_get_alias_unset(self):
        self.assertRaises(NotImplementedError, _EmptyAPIDefinition.get_alias)

    def test_get_description(self):
        self.assertEqual(self.DESCRIPTION, self.extn.get_description())

    def test_get_description_unset(self):
        self.assertRaises(NotImplementedError, _EmptyAPIDefinition.get_description)

    def test_get_updated(self):
        self.assertEqual(self.UPDATED_TIMESTAMP, self.extn.get_updated())

    def test_get_updated_unset(self):
        self.assertRaises(NotImplementedError, _EmptyAPIDefinition.get_updated)

    def test_get_extended_resources_v2(self):
        self.assertEqual(dict(list(self.RESOURCE_ATTRIBUTE_MAP.items()) + list(self.SUB_RESOURCE_ATTRIBUTE_MAP.items())), self.extn.get_extended_resources('2.0'))

    def test_get_extended_resources_v2_unset(self):
        self.assertRaises(NotImplementedError, self.empty_extn.get_extended_resources, '2.0')

    def test_get_extended_resources_v1(self):
        self.assertEqual({}, self.extn.get_extended_resources('1.0'))

    def test_get_extended_resources_v1_unset(self):
        self.assertEqual({}, self.empty_extn.get_extended_resources('1.0'))

    def test_get_required_extensions(self):
        self.assertEqual(self.REQUIRED_EXTENSIONS, self.extn.get_required_extensions())

    def test_get_required_extensions_unset(self):
        self.assertRaises(NotImplementedError, self.empty_extn.get_required_extensions)

    def test_get_optional_extensions(self):
        self.assertEqual(self.OPTIONAL_EXTENSIONS, self.extn.get_optional_extensions())

    def test_get_optional_extensions_unset(self):
        self.assertRaises(NotImplementedError, self.empty_extn.get_optional_extensions)

    def test_update_attributes_map_extensions_unset(self):
        self.assertRaises(NotImplementedError, self.empty_extn.update_attributes_map, {})

    def test_update_attributes_map_with_ext_attrs(self):
        base_attrs = {'ports': {'a': 'A'}}
        ext_attrs = {'ports': {'b': 'B'}}
        self.extn.update_attributes_map(base_attrs, ext_attrs)
        self.assertEqual({'ports': {'a': 'A', 'b': 'B'}}, ext_attrs)

    def test_update_attributes_map_without_ext_attrs(self):
        base_attrs = {'ports': {'a': 'A'}}
        self.extn.update_attributes_map(base_attrs)
        self.assertIn('a', self.extn.get_extended_resources('2.0')['ports'])