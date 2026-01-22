import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
class ValidatorTestCase(base.PolicyBaseTestCase):

    def _get_test_enforcer(self):
        test_rules = [policy.RuleDefault('foo', 'foo:bar=baz'), policy.RuleDefault('bar', 'bar:foo=baz')]
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults(test_rules)
        return enforcer

    def _test_policy(self, rule, success=False, missing_file=False):
        policy_file = self.get_config_file_fullname('test.yaml')
        if missing_file:
            policy_file = 'bogus.yaml'
        self.create_config_file('test.yaml', rule)
        self.create_config_file('test.conf', '[oslo_policy]\npolicy_file=%s' % policy_file)
        self.conf(args=['--config-dir', self.config_dir])
        with mock.patch('oslo_policy.generator._get_enforcer') as ge:
            ge.return_value = self._get_test_enforcer()
            result = generator._validate_policy('test')
            if success:
                self.assertEqual(0, result)
            else:
                self.assertEqual(1, result)

    def test_success(self):
        self._test_policy('foo: rule:bar', success=True)

    def test_cyclical_reference(self):
        self._test_policy('foo: rule:bar\nbar: rule:foo')

    def test_invalid_syntax(self):
        self._test_policy('foo: (bar))')

    def test_false_okay(self):
        self._test_policy('foo: !', success=True)

    def test_reference_nonexistent(self):
        self._test_policy('foo: rule:baz')

    def test_nonexistent(self):
        self._test_policy('baz: rule:foo')

    def test_missing_policy_file(self):
        self._test_policy('', missing_file=True)