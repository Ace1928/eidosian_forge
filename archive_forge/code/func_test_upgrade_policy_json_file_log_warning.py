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
@mock.patch.object(generator, 'LOG')
def test_upgrade_policy_json_file_log_warning(self, mock_log):
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=self.extensions, namespace='test_upgrade')
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
        testargs = ['olsopolicy-policy-upgrade', '--policy', self.get_config_file_fullname('policy.json'), '--namespace', 'test_upgrade', '--output-file', self.get_config_file_fullname('new_policy.json'), '--format', 'json']
        with mock.patch('sys.argv', testargs):
            generator.upgrade_policy(conf=self.local_conf)
            mock_log.warning.assert_any_call(policy.WARN_JSON)