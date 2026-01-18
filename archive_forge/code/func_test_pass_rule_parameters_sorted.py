import copy
from unittest import mock
from oslo_serialization import jsonutils
from oslo_policy import shell
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
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