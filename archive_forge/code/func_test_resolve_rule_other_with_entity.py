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
def test_resolve_rule_other_with_entity(self):
    client_plugin, schema = self._test_resolve_rule()
    data = {'far': 'one'}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.RESOLVE, ['far'], client_plugin=client_plugin, finder='find_name_id', entity='rose')
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far'))
    result = tran.translate('far', data['far'])
    self.assertEqual('pink', result)
    self.assertEqual('pink', tran.resolved_translations['far'])