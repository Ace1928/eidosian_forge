from unittest import mock
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import quotas as osc_quotas
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_quota_set_update_project_exception(self):
    arglist = [self.project.id, '--share-groups', '40', '--share-group-snapshots', '40']
    verifylist = [('project', self.project.id), ('share_groups', 40), ('share_group_snapshots', 40)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.quotas_mock.update.side_effect = BadRequest()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)