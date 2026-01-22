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
class GeneratePolicyTestCase(base.PolicyBaseTestCase):

    def setUp(self):
        super(GeneratePolicyTestCase, self).setUp()

    def test_merged_rules(self):
        extensions = []
        for name, opts in OPTS.items():
            ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
            extensions.append(ext)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['base_rules', 'rules'])
        sample_file = self.get_config_file_fullname('policy-sample.yaml')
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
            generator._generate_sample(['base_rules', 'rules'], sample_file, include_help=False)
        enforcer = policy.Enforcer(self.conf, policy_file='policy-sample.yaml')
        enforcer.register_default(policy.RuleDefault('admin', 'is_admin:False'))
        enforcer.register_default(policy.RuleDefault('foo', 'role:foo'))
        ext = stevedore.extension.Extension(name='testing', entry_point=None, plugin=None, obj=enforcer)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=[ext], namespace='testing')
        merged_file = self.get_config_file_fullname('policy-merged.yaml')
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
            generator._generate_policy(namespace='testing', output_file=merged_file)
            mock_ext_mgr.assert_called_once_with('oslo.policy.enforcer', names=['testing'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
        merged_enforcer = policy.Enforcer(self.conf, policy_file='policy-merged.yaml')
        merged_enforcer.load_rules()
        for rule in ['admin', 'owner', 'admin_or_owner', 'foo']:
            self.assertIn(rule, merged_enforcer.rules)
        self.assertEqual('is_admin:True', str(merged_enforcer.rules['admin']))
        self.assertEqual('role:foo', str(merged_enforcer.rules['foo']))