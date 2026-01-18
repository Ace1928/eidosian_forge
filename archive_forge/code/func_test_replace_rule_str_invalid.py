import copy
from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import parameters
from heat.engine import properties
from heat.engine import translation
from heat.tests import common
def test_replace_rule_str_invalid(self):
    schema = {'far': properties.Schema(properties.Schema.STRING), 'bar': properties.Schema(properties.Schema.INTEGER)}
    data = {'far': 'one', 'bar': 2}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.REPLACE, ['bar'], props.get('far'))
    props.update_translation([rule])
    exc = self.assertRaises(exception.StackValidationFailed, props.validate)
    self.assertEqual("Property error: bar: Value 'one' is not an integer", str(exc))