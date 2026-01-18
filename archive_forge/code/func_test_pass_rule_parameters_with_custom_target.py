import copy
from unittest import mock
from oslo_serialization import jsonutils
from oslo_policy import shell
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
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