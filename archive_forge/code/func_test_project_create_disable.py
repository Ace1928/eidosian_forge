from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_project_create_disable(self):
    arglist = ['--disable', self.fake_project.name]
    verifylist = [('enable', False), ('disable', True), ('name', self.fake_project.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'description': None, 'enabled': False}
    self.projects_mock.create.assert_called_with(self.fake_project.name, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)