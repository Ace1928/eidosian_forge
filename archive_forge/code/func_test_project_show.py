from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_project_show(self):
    arglist = [self.fake_proj_show.id]
    verifylist = [('project', self.fake_proj_show.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.get.assert_called_with(self.fake_proj_show.id)
    collist = ('description', 'enabled', 'id', 'name', 'properties')
    self.assertEqual(collist, columns)
    datalist = (self.fake_proj_show.description, True, self.fake_proj_show.id, self.fake_proj_show.name, format_columns.DictColumn({}))
    self.assertCountEqual(datalist, data)