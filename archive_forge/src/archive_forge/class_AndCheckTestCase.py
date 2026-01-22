from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class AndCheckTestCase(test_base.BaseTestCase):

    def test_init(self):
        check = _checks.AndCheck(['rule1', 'rule2'])
        self.assertEqual(['rule1', 'rule2'], check.rules)

    def test_add_check(self):
        check = _checks.AndCheck(['rule1', 'rule2'])
        check.add_check('rule3')
        self.assertEqual(['rule1', 'rule2', 'rule3'], check.rules)

    def test_str(self):
        check = _checks.AndCheck(['rule1', 'rule2'])
        self.assertEqual('(rule1 and rule2)', str(check))

    def test_call_all_false(self):
        rules = [_BoolCheck(False), _BoolCheck(False)]
        check = _checks.AndCheck(rules)
        self.assertFalse(check('target', 'cred', None))
        self.assertTrue(rules[0].called)
        self.assertFalse(rules[1].called)

    def test_call_first_true(self):
        rules = [_BoolCheck(True), _BoolCheck(False)]
        check = _checks.AndCheck(rules)
        self.assertFalse(check('target', 'cred', None))
        self.assertTrue(rules[0].called)
        self.assertTrue(rules[1].called)

    def test_call_second_true(self):
        rules = [_BoolCheck(False), _BoolCheck(True)]
        check = _checks.AndCheck(rules)
        self.assertFalse(check('target', 'cred', None))
        self.assertTrue(rules[0].called)
        self.assertFalse(rules[1].called)

    def test_rule_takes_current_rule(self):
        results = []

        class TestCheck(object):

            def __call__(self, target, cred, enforcer, current_rule=None):
                results.append((target, cred, enforcer, current_rule))
                return False
        check = _checks.AndCheck([TestCheck()])
        self.assertFalse(check('target', 'cred', None, current_rule='a_rule'))
        self.assertEqual([('target', 'cred', None, 'a_rule')], results)

    def test_rule_does_not_take_current_rule(self):
        results = []

        class TestCheck(object):

            def __call__(self, target, cred, enforcer):
                results.append((target, cred, enforcer))
                return False
        check = _checks.AndCheck([TestCheck()])
        self.assertFalse(check('target', 'cred', None, current_rule='a_rule'))
        self.assertEqual([('target', 'cred', None)], results)