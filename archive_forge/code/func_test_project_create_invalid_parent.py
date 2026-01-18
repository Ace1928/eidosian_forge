from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_create_invalid_parent(self):
    self.projects_mock.resource_class.__name__ = 'Project'
    self.projects_mock.get.side_effect = exceptions.NotFound('Invalid parent')
    self.projects_mock.find.side_effect = exceptions.NotFound('Invalid parent')
    arglist = ['--domain', self.project.domain_id, '--parent', 'invalid', self.project.name]
    verifylist = [('domain', self.project.domain_id), ('parent', 'invalid'), ('enable', False), ('disable', False), ('name', self.project.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)