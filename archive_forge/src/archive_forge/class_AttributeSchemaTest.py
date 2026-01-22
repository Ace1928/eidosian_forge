from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
class AttributeSchemaTest(common.HeatTestCase):

    def test_schema_all(self):
        d = {'description': 'A attribute'}
        s = attributes.Schema('A attribute')
        self.assertEqual(d, dict(s))
        d = {'description': 'Another attribute', 'type': 'string'}
        s = attributes.Schema('Another attribute', type=attributes.Schema.STRING)
        self.assertEqual(d, dict(s))

    def test_all_resource_schemata(self):
        for resource_type in resources.global_env().get_types():
            for schema in getattr(resource_type, 'attributes_schema', {}).values():
                attributes.Schema.from_attribute(schema)

    def test_from_attribute_new_schema_format(self):
        s = attributes.Schema('Test description.')
        self.assertIs(s, attributes.Schema.from_attribute(s))
        self.assertEqual('Test description.', attributes.Schema.from_attribute(s).description)
        s = attributes.Schema('Test description.', type=attributes.Schema.MAP)
        self.assertIs(s, attributes.Schema.from_attribute(s))
        self.assertEqual(attributes.Schema.MAP, attributes.Schema.from_attribute(s).type)

    def test_schema_support_status(self):
        schema = {'foo_sup': attributes.Schema('Description1'), 'bar_dep': attributes.Schema('Description2', support_status=support.SupportStatus(support.DEPRECATED, 'Do not use this ever'))}
        attrs = attributes.Attributes('test_rsrc', schema, lambda d: d)
        self.assertEqual(support.SUPPORTED, attrs._attributes['foo_sup'].support_status().status)
        self.assertEqual(support.DEPRECATED, attrs._attributes['bar_dep'].support_status().status)
        self.assertEqual('Do not use this ever', attrs._attributes['bar_dep'].support_status().message)