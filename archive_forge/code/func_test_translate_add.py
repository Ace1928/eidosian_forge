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
def test_translate_add(self):
    rules = [translation.TranslationRule(self.props, translation.TranslationRule.ADD, ['a', 'b'], value=['check'])]
    tran = translation.Translation()
    tran.set_rules(rules)
    result = tran.translate('a.b', ['test'])
    self.assertEqual(['test', 'check'], result)
    self.assertEqual(['test', 'check'], tran.resolved_translations['a.b'])
    tran.resolved_translations = {}
    tran.store_translated_values = False
    result = tran.translate('a.b', ['test'])
    self.assertEqual(['test', 'check'], result)
    self.assertEqual({}, tran.resolved_translations)
    tran.store_translated_values = True
    self.assertEqual(['check'], tran.translate('a.b', None))
    self.assertEqual(['test', 'check'], tran.translate('a.0.b', ['test']))