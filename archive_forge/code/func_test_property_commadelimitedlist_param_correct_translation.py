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
def test_property_commadelimitedlist_param_correct_translation(self):
    """Test when property with sub-schema takes comma_delimited_list."""
    schema = {'far': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.STRING)), 'boo': properties.Schema(properties.Schema.STRING)}

    class DummyStack(dict):

        @property
        def parameters(self):
            return mock.Mock()
    param = hot_funcs.GetParam(DummyStack(list_far='list_far'), 'get_param', 'list_far')
    param.parameters = {'list_far': parameters.CommaDelimitedListParam('list_far', {'Type': 'CommaDelimitedList'}, 'white,roses').value()}
    data = {'far': param, 'boo': 'chrysanthemums'}
    props = properties.Properties(schema, data, resolver=function.resolve)
    rule = translation.TranslationRule(props, translation.TranslationRule.ADD, ['far'], [props.get('boo')])
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far'))
    result = tran.translate('far', props['far'])
    self.assertEqual(['white', 'roses', 'chrysanthemums'], result)
    self.assertEqual(['white', 'roses', 'chrysanthemums'], tran.resolved_translations['far'])