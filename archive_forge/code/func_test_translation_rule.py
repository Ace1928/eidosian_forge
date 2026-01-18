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
def test_translation_rule(self):
    for r in translation.TranslationRule.RULE_KEYS:
        props = properties.Properties({}, {})
        rule = translation.TranslationRule(props, r, ['any'], ['value'] if r == 'Add' else None, 'value_name' if r == 'Replace' else None, 'client_plugin' if r == 'Resolve' else None, 'finder' if r == 'Resolve' else None)
        self.assertEqual(rule.properties, props)
        self.assertEqual(rule.rule, r)
        if r == 'Add':
            self.assertEqual(['value'], rule.value)
        if r == 'Replace':
            self.assertEqual('value_name', rule.value_name)
        else:
            self.assertIsNone(rule.value_name)