from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_project_set_unexist_project(self):
    arglist = ['unexist-project']
    verifylist = [('project', 'unexist-project'), ('name', None), ('description', None), ('enable', False), ('disable', False), ('property', None)]
    self.projects_mock.get.side_effect = exceptions.NotFound(None)
    self.projects_mock.find.side_effect = exceptions.NotFound(None)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)