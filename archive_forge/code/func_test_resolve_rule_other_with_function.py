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
def test_resolve_rule_other_with_function(self):
    client_plugin, schema = self._test_resolve_rule()
    join_func = cfn_funcs.Join(None, 'Fn::Join', ['.', ['bar', 'baz']])
    data = {'far': join_func}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.RESOLVE, ['far'], client_plugin=client_plugin, finder='find_name_id')
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far'))
    result = tran.translate('far', data['far'])
    self.assertEqual('yellow', result)
    self.assertEqual('yellow', tran.resolved_translations['far'])