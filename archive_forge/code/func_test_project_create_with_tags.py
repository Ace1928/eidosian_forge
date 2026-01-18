from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_create_with_tags(self):
    arglist = ['--domain', self.project.domain_id, '--tag', 'foo', self.project.name]
    verifylist = [('domain', self.project.domain_id), ('enable', False), ('disable', False), ('name', self.project.name), ('parent', None), ('tags', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'name': self.project.name, 'domain': self.project.domain_id, 'description': None, 'enabled': True, 'parent': None, 'tags': ['foo'], 'options': {}}
    self.projects_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)