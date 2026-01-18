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
def test_translate_delete(self):
    rules = [translation.TranslationRule(self.props, translation.TranslationRule.DELETE, ['a'])]
    tran = translation.Translation()
    tran.set_rules(rules)
    self.assertIsNone(tran.translate('a'))
    self.assertIsNone(tran.resolved_translations['a'])
    tran.resolved_translations = {}
    tran.store_translated_values = False
    self.assertIsNone(tran.translate('a'))
    self.assertEqual({}, tran.resolved_translations)
    tran.store_translated_values = True