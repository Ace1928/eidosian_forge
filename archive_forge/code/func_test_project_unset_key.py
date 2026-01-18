from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_project_unset_key(self):
    arglist = ['--property', 'fee', '--property', 'fo', self.fake_proj.name]
    verifylist = [('property', ['fee', 'fo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'description': self.fake_proj.description, 'enabled': True, 'fee': None, 'fo': None, 'id': self.fake_proj.id, 'name': self.fake_proj.name}
    self.projects_mock.update.assert_called_with(self.fake_proj.id, **kwargs)
    self.assertIsNone(result)