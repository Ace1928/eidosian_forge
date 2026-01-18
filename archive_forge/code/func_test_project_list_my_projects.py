from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_list_my_projects(self):
    auth_ref = identity_fakes.fake_auth_ref(identity_fakes.TOKEN_WITH_PROJECT_ID)
    ar_mock = mock.PropertyMock(return_value=auth_ref)
    type(self.app.client_manager).auth_ref = ar_mock
    arglist = ['--my-projects']
    verifylist = [('my_projects', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.list.assert_called_with(user=self.app.client_manager.auth_ref.user_id)
    collist = ('ID', 'Name')
    self.assertEqual(collist, columns)
    datalist = ((self.project.id, self.project.name),)
    self.assertEqual(datalist, tuple(data))