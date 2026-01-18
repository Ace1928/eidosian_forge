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
def test_replace_rule_map_with_custom_value_path(self):
    schema = {'far': properties.Schema(properties.Schema.MAP, schema={'red': properties.Schema(properties.Schema.STRING)}), 'bar': properties.Schema(properties.Schema.MAP)}
    data = {'far': {}, 'bar': {'red': 'dak'}}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.REPLACE, ['far', 'red'], value_path=['bar'], custom_value_path=['red'])
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far.red'))
    result = tran.translate('far.red')
    self.assertEqual('dak', result)
    self.assertEqual('dak', tran.resolved_translations['far.red'])