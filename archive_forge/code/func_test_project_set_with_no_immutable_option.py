from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_set_with_no_immutable_option(self):
    arglist = ['--domain', self.project.domain_id, '--no-immutable', self.project.name]
    verifylist = [('domain', self.project.domain_id), ('no_immutable', True), ('enable', False), ('disable', False), ('project', self.project.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'options': {'immutable': False}}
    self.projects_mock.update.assert_called_with(self.project.id, **kwargs)
    self.assertIsNone(result)