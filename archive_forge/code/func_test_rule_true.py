from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_rule_true(self):
    self.enforcer.rules = dict(spam=_BoolCheck(True))
    check = _checks.RuleCheck('rule', 'spam')
    self.assertTrue(check('target', 'creds', self.enforcer))