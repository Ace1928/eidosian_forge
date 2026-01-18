from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
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