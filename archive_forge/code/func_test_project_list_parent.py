from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_list_parent(self):
    self.parent = identity_fakes.FakeProject.create_one_project()
    self.project = identity_fakes.FakeProject.create_one_project(attrs={'domain_id': self.domain.id, 'parent_id': self.parent.id})
    arglist = ['--parent', self.parent.id]
    verifylist = [('parent', self.parent.id)]
    self.projects_mock.get.return_value = self.parent
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.list.assert_called_with(parent=self.parent.id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))