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
def test_property_json_param_to_list_correct_translation(self):
    """Test case when list property with sub-schema takes json param."""
    schema = {'far': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema={'bar': properties.Schema(properties.Schema.STRING), 'dar': properties.Schema(properties.Schema.STRING)}))}

    class DummyStack(dict):

        @property
        def parameters(self):
            return mock.Mock()
    param = hot_funcs.GetParam(DummyStack(json_far='json_far'), 'get_param', 'json_far')
    param.parameters = {'json_far': parameters.JsonParam('json_far', {'Type': 'Json'}, '{"dar": "rad"}').value()}
    data = {'far': [param]}
    props = properties.Properties(schema, data, resolver=function.resolve)
    rule = translation.TranslationRule(props, translation.TranslationRule.REPLACE, ['far', 'bar'], value_name='dar')
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far.0.bar'))
    prop_data = props['far']
    result = tran.translate('far.0.bar', prop_data[0]['bar'], prop_data[0])
    self.assertEqual('rad', result)
    self.assertEqual('rad', tran.resolved_translations['far.0.bar'])