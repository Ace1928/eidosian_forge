from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_remove_tags(self):
    arglist = ['--remove-tag', 'tag1', '--remove-tag', 'tag2', self.project.name]
    verifylist = [('enable', False), ('disable', False), ('project', self.project.name), ('remove_tag', ['tag1', 'tag2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'tags': list(set(['tag3']))}
    self.projects_mock.update.assert_called_with(self.project.id, **kwargs)
    self.assertIsNone(result)