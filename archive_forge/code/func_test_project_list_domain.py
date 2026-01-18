from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_list_domain(self):
    arglist = ['--domain', self.project.domain_id]
    verifylist = [('domain', self.project.domain_id)]
    self.domains_mock.get.return_value = self.domain
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.list.assert_called_with(domain=self.project.domain_id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))