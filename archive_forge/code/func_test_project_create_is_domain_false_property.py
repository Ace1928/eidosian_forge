from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_create_is_domain_false_property(self):
    arglist = ['--property', 'is_domain=false', self.project.name]
    verifylist = [('parent', None), ('enable', False), ('disable', False), ('name', self.project.name), ('tags', []), ('property', {'is_domain': 'false'}), ('name', self.project.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'name': self.project.name, 'domain': None, 'description': None, 'enabled': True, 'parent': None, 'is_domain': False, 'tags': [], 'options': {}}
    self.projects_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)