from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
class AttributesTypeTest(common.HeatTestCase):
    scenarios = [('string_type', dict(a_type=attributes.Schema.STRING, value='correct value', invalid_value=[])), ('list_type', dict(a_type=attributes.Schema.LIST, value=[], invalid_value='invalid_value')), ('map_type', dict(a_type=attributes.Schema.MAP, value={}, invalid_value='invalid_value')), ('integer_type', dict(a_type=attributes.Schema.INTEGER, value=1, invalid_value='invalid_value')), ('boolean_type', dict(a_type=attributes.Schema.BOOLEAN, value=True, invalid_value='invalid_value')), ('boolean_type_string_true', dict(a_type=attributes.Schema.BOOLEAN, value='True', invalid_value='invalid_value')), ('boolean_type_string_false', dict(a_type=attributes.Schema.BOOLEAN, value='false', invalid_value='invalid_value'))]

    def test_validate_type(self):
        resolver = mock.Mock()
        msg = 'Attribute test1 is not of type %s' % self.a_type
        attr_schema = attributes.Schema('Test attribute', type=self.a_type)
        attrs_schema = {'res1': attr_schema}
        attr = attributes.Attribute('test1', attr_schema)
        attribs = attributes.Attributes('test res1', attrs_schema, resolver)
        attribs._validate_type(attr, self.value)
        self.assertNotIn(msg, self.LOG.output)
        attribs._validate_type(attr, self.invalid_value)
        self.assertIn(msg, self.LOG.output)