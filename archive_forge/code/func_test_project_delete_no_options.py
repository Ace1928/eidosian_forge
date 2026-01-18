from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_project_delete_no_options(self):
    arglist = [self.fake_project.id]
    verifylist = [('projects', [self.fake_project.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.projects_mock.delete.assert_called_with(self.fake_project.id)
    self.assertIsNone(result)