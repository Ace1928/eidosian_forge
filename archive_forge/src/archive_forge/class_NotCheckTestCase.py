from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class NotCheckTestCase(test_base.BaseTestCase):

    def test_init(self):
        check = _checks.NotCheck('rule')
        self.assertEqual('rule', check.rule)

    def test_str(self):
        check = _checks.NotCheck('rule')
        self.assertEqual('not rule', str(check))

    def test_call_true(self):
        rule = _checks.TrueCheck()
        check = _checks.NotCheck(rule)
        self.assertFalse(check('target', 'cred', None))

    def test_call_false(self):
        rule = _checks.FalseCheck()
        check = _checks.NotCheck(rule)
        self.assertTrue(check('target', 'cred', None))

    def test_rule_takes_current_rule(self):
        results = []

        class TestCheck(object):

            def __call__(self, target, cred, enforcer, current_rule=None):
                results.append((target, cred, enforcer, current_rule))
                return True
        check = _checks.NotCheck(TestCheck())
        self.assertFalse(check('target', 'cred', None, current_rule='a_rule'))
        self.assertEqual([('target', 'cred', None, 'a_rule')], results)

    def test_rule_does_not_take_current_rule(self):
        results = []

        class TestCheck(object):

            def __call__(self, target, cred, enforcer):
                results.append((target, cred, enforcer))
                return True
        check = _checks.NotCheck(TestCheck())
        self.assertFalse(check('target', 'cred', None, current_rule='a_rule'))
        self.assertEqual([('target', 'cred', None)], results)