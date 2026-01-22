import copy
from unittest import mock
from oslo_serialization import jsonutils
from oslo_policy import shell
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class CheckerTestCase(base.PolicyBaseTestCase):
    SAMPLE_POLICY = '---\n"sample_rule": "role:service"\n"sampleservice:sample_rule": ""\n'
    SAMPLE_POLICY_UNSORTED = '---\n"sample_rule": "role:service"\n"sampleservice:sample_rule2": ""\n"sampleservice:sample_rule0": ""\n"sampleservice:sample_rule1": ""\n'
    SAMPLE_POLICY_SCOPED = '---\n"sampleservice:sample_rule": "role:role1"\n"sampleservice:scoped_rule": "role:role1 and system_scope:all"\n'
    SAMPLE_POLICY_OWNER = '---\n"sampleservice:owner_rule": "user_id:%(user_id)s"\n'

    def setUp(self):
        super(CheckerTestCase, self).setUp()
        self.create_config_file('policy.yaml', self.SAMPLE_POLICY)
        self.create_config_file('access.json', jsonutils.dumps(token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE))

    @mock.patch('oslo_policy._checks.TrueCheck.__call__')
    def test_pass_rule_parameters(self, call_mock):
        policy_file = self.get_config_file_fullname('policy.yaml')
        access_file = self.get_config_file_fullname('access.json')
        apply_rule = None
        is_admin = False
        stdout = self._capture_stdout()
        access_data = copy.deepcopy(token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE['token'])
        target = {'user_id': access_data['user']['id'], 'project_id': access_data['project']['id']}
        access_data['roles'] = [role['name'] for role in access_data['roles']]
        access_data['user_id'] = access_data['user']['id']
        access_data['project_id'] = access_data['project']['id']
        access_data['is_admin'] = is_admin
        shell.tool(policy_file, access_file, apply_rule, is_admin)
        call_mock.assert_called_once_with(target, access_data, mock.ANY, current_rule='sampleservice:sample_rule')
        expected = 'passed: sampleservice:sample_rule\n'
        self.assertEqual(expected, stdout.getvalue())

    def test_pass_rule_parameters_with_scope(self):
        self.create_config_file('policy.yaml', self.SAMPLE_POLICY_SCOPED)
        self.create_config_file('access.json', jsonutils.dumps(token_fixture.SYSTEM_SCOPED_TOKEN_FIXTURE))
        policy_file = self.get_config_file_fullname('policy.yaml')
        access_file = self.get_config_file_fullname('access.json')
        apply_rule = None
        is_admin = False
        stdout = self._capture_stdout()
        access_data = copy.deepcopy(token_fixture.SYSTEM_SCOPED_TOKEN_FIXTURE['token'])
        access_data['roles'] = [role['name'] for role in access_data['roles']]
        access_data['user_id'] = access_data['user']['id']
        access_data['is_admin'] = is_admin
        shell.tool(policy_file, access_file, apply_rule, is_admin)
        expected = 'passed: sampleservice:sample_rule\npassed: sampleservice:scoped_rule\n'
        self.assertEqual(expected, stdout.getvalue())

    def test_pass_rule_parameters_with_owner(self):
        self.create_config_file('policy.yaml', self.SAMPLE_POLICY_OWNER)
        self.create_config_file('access.json', jsonutils.dumps(token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE))
        policy_file = self.get_config_file_fullname('policy.yaml')
        access_file = self.get_config_file_fullname('access.json')
        apply_rule = None
        is_admin = False
        stdout = self._capture_stdout()
        access_data = copy.deepcopy(token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE['token'])
        access_data['roles'] = [role['name'] for role in access_data['roles']]
        access_data['user_id'] = access_data['user']['id']
        access_data['project_id'] = access_data['project']['id']
        access_data['is_admin'] = is_admin
        shell.tool(policy_file, access_file, apply_rule, is_admin)
        expected = 'passed: sampleservice:owner_rule\n'
        self.assertEqual(expected, stdout.getvalue())

    def test_pass_rule_parameters_sorted(self):
        self.create_config_file('policy.yaml', self.SAMPLE_POLICY_UNSORTED)
        policy_file = self.get_config_file_fullname('policy.yaml')
        access_file = self.get_config_file_fullname('access.json')
        apply_rule = None
        is_admin = False
        stdout = self._capture_stdout()
        access_data = copy.deepcopy(token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE['token'])
        access_data['roles'] = [role['name'] for role in access_data['roles']]
        access_data['user_id'] = access_data['user']['id']
        access_data['project_id'] = access_data['project']['id']
        access_data['is_admin'] = is_admin
        shell.tool(policy_file, access_file, apply_rule, is_admin)
        expected = 'passed: sampleservice:sample_rule0\npassed: sampleservice:sample_rule1\npassed: sampleservice:sample_rule2\n'
        self.assertEqual(expected, stdout.getvalue())

    @mock.patch('oslo_policy._checks.TrueCheck.__call__')
    def test_pass_rule_parameters_with_custom_target(self, call_mock):
        apply_rule = None
        is_admin = False
        access_data = copy.deepcopy(token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE['token'])
        access_data['roles'] = [role['name'] for role in access_data['roles']]
        access_data['user_id'] = access_data['user']['id']
        access_data['project_id'] = access_data['project']['id']
        access_data['is_admin'] = is_admin
        sample_target = {'project_id': access_data['project']['id'], 'domain_id': access_data['project']['domain']['id']}
        self.create_config_file('target.json', jsonutils.dumps(sample_target))
        policy_file = self.get_config_file_fullname('policy.yaml')
        access_file = self.get_config_file_fullname('access.json')
        target_file = self.get_config_file_fullname('target.json')
        stdout = self._capture_stdout()
        shell.tool(policy_file, access_file, apply_rule, is_admin, target_file)
        call_mock.assert_called_once_with(sample_target, access_data, mock.ANY, current_rule='sampleservice:sample_rule')
        expected = 'passed: sampleservice:sample_rule\n'
        self.assertEqual(expected, stdout.getvalue())

    def test_all_nonadmin(self):
        policy_file = self.get_config_file_fullname('policy.yaml')
        access_file = self.get_config_file_fullname('access.json')
        apply_rule = None
        is_admin = False
        stdout = self._capture_stdout()
        shell.tool(policy_file, access_file, apply_rule, is_admin)
        expected = 'passed: sampleservice:sample_rule\n'
        self.assertEqual(expected, stdout.getvalue())

    def test_flatten_from_dict(self):
        target = {'target': {'secret': {'project_id': '1234'}}}
        result = shell.flatten(target)
        self.assertEqual(result, {'target.secret.project_id': '1234'})

    def test_flatten_from_file(self):
        target = {'target': {'secret': {'project_id': '1234'}}}
        self.create_config_file('target.json', jsonutils.dumps(target))
        with open(self.get_config_file_fullname('target.json'), 'r') as fh:
            target_from_file = fh.read()
        result = shell.flatten(jsonutils.loads(target_from_file))
        self.assertEqual(result, {'target.secret.project_id': '1234'})