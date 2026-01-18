import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_trunk_attrs_with_exception(self):
    arglist = ['--name', 'reallylongname', self._trunk['name']]
    verifylist = [('trunk', self._trunk['name']), ('name', 'reallylongname')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.network_client.update_trunk = mock.Mock(side_effect=exceptions.CommandError)
    with testtools.ExpectedException(exceptions.CommandError) as e:
        self.cmd.take_action(parsed_args)
        self.assertEqual("Failed to set trunk '%s': " % self._trunk['name'], str(e))
    attrs = {'name': 'reallylongname'}
    self.network_client.update_trunk.assert_called_once_with(self._trunk, **attrs)
    self.network_client.add_trunk_subports.assert_not_called()