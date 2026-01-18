import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
@mock.patch('warnings.warn', new=mock.Mock())
def test_override_deprecated_policy_with_new_rule(self):
    rules = jsonutils.dumps({'old_rule': 'rule:new_rule'})
    self.create_config_file('policy.json', rules)
    deprecated_rule = policy.DeprecatedRule(name='old_rule', check_str='role:bang', deprecated_reason='"old_rule" is a bad name', deprecated_since='N')
    rule_list = [policy.DocumentedRuleDefault(name='new_rule', check_str='role:bang', description='Replacement for old_rule.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
    self.enforcer.register_defaults(rule_list)
    self.assertFalse(self.enforcer.enforce('new_rule', {}, {'roles': ['fizz']}))
    self.assertTrue(self.enforcer.enforce('new_rule', {}, {'roles': ['bang']}))
    self.assertEqual('bang', self.enforcer.rules['new_rule'].match)