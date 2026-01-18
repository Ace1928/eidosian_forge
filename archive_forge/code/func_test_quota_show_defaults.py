from unittest import mock
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import quotas as osc_quotas
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_quota_show_defaults(self):
    arglist = [self.project.id, '--defaults']
    verifylist = [('project', self.project.id), ('defaults', True)]
    self.quotas_mock.defaults = mock.Mock()
    self.quotas_mock.defaults.return_value = self.quotas
    with mock.patch('osc_lib.utils.find_resource') as mock_find_resource:
        mock_find_resource.return_value = self.project
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.quotas_mock.defaults.assert_called_with(self.project.id)
        self.assertCountEqual(columns, self.quotas.keys())
        self.assertCountEqual(data, self.quotas._info.values())