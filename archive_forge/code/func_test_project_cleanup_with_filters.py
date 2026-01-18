from io import StringIO
from unittest import mock
from openstackclient.common import project_cleanup
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_project_cleanup_with_filters(self):
    arglist = ['--project', self.project.id, '--created-before', '2200-01-01', '--updated-before', '2200-01-02']
    verifylist = [('dry_run', False), ('auth_project', False), ('project', self.project.id), ('created_before', '2200-01-01'), ('updated_before', '2200-01-02')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = None
    with mock.patch('sys.stdin', StringIO('y')):
        result = self.cmd.take_action(parsed_args)
    self.sdk_connect_as_project_mock.assert_called_with(self.project)
    filters = {'created_at': '2200-01-01', 'updated_at': '2200-01-02'}
    calls = [mock.call(dry_run=True, status_queue=mock.ANY, filters=filters, skip_resources=None), mock.call(dry_run=False, status_queue=mock.ANY, filters=filters)]
    self.project_cleanup_mock.assert_has_calls(calls)
    self.assertIsNone(result)