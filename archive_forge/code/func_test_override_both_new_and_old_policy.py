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
def test_override_both_new_and_old_policy(self):
    rules_dict = {'foo:create_bar': 'role:bazz', 'foo:bar': 'role:wee'}
    rules = jsonutils.dumps(rules_dict)
    self.create_config_file('policy.json', rules)
    deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
    rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
    self.enforcer.register_defaults(rule_list)
    self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}))
    self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}))
    self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['wee']}))
    self.assertTrue(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['bazz']}))