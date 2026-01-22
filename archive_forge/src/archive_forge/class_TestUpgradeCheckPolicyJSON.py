import fixtures
import os.path
import tempfile
import yaml
from oslo_config import cfg
from oslo_config import fixture as config
from oslo_policy import opts as policy_opts
from oslo_serialization import jsonutils
from oslotest import base
from oslo_upgradecheck import common_checks
from oslo_upgradecheck import upgradecheck
class TestUpgradeCheckPolicyJSON(base.BaseTestCase):

    def setUp(self):
        super(TestUpgradeCheckPolicyJSON, self).setUp()
        conf_fixture = self.useFixture(config.Config())
        conf_fixture.load_raw_values()
        self.conf = conf_fixture.conf
        self.conf.register_opts(policy_opts._options, group=policy_opts._option_group)
        self.cmd = upgradecheck.UpgradeCommands()
        self.cmd._upgrade_checks = (('Policy File JSON to YAML Migration', (common_checks.check_policy_json, {'conf': self.conf})),)
        self.data = {'rule_admin': 'True', 'rule_admin2': 'is_admin:True'}
        self.temp_dir = self.useFixture(fixtures.TempDir())
        fd, self.json_file = tempfile.mkstemp(dir=self.temp_dir.path)
        fd, self.yaml_file = tempfile.mkstemp(dir=self.temp_dir.path)
        with open(self.json_file, 'w') as fh:
            jsonutils.dump(self.data, fh)
        with open(self.yaml_file, 'w') as fh:
            yaml.dump(self.data, fh)
        original_search_dirs = cfg._search_dirs

        def fake_search_dirs(dirs, name):
            dirs.append(self.temp_dir.path)
            return original_search_dirs(dirs, name)
        mock_search_dir = self.useFixture(fixtures.MockPatch('oslo_config.cfg._search_dirs')).mock
        mock_search_dir.side_effect = fake_search_dirs

    def test_policy_json_file_fail_upgrade(self):
        self.conf.set_override('policy_file', self.json_file, group='oslo_policy')
        self.assertEqual(upgradecheck.Code.FAILURE, self.cmd.check())

    def test_policy_yaml_file_pass_upgrade(self):
        self.conf.set_override('policy_file', self.yaml_file, group='oslo_policy')
        self.assertEqual(upgradecheck.Code.SUCCESS, self.cmd.check())

    def test_no_policy_file_pass_upgrade(self):
        self.conf.set_override('policy_file', 'non_exist_file', group='oslo_policy')
        self.assertEqual(upgradecheck.Code.SUCCESS, self.cmd.check())

    def test_default_policy_yaml_file_pass_upgrade(self):
        self.conf.set_override('policy_file', 'policy.yaml', group='oslo_policy')
        tmpfilename = os.path.join(self.temp_dir.path, 'policy.yaml')
        with open(tmpfilename, 'w') as fh:
            yaml.dump(self.data, fh)
        self.assertEqual(upgradecheck.Code.SUCCESS, self.cmd.check())

    def test_old_default_policy_json_file_fail_upgrade(self):
        self.conf.set_override('policy_file', 'policy.json', group='oslo_policy')
        tmpfilename = os.path.join(self.temp_dir.path, 'policy.json')
        with open(tmpfilename, 'w') as fh:
            jsonutils.dump(self.data, fh)
        self.assertEqual(upgradecheck.Code.FAILURE, self.cmd.check())