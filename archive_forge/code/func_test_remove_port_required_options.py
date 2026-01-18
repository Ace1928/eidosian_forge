from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_remove_port_required_options(self):
    arglist = [self._router.id, self._router.port]
    verifylist = [('router', self._router.id), ('port', self._router.port)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.remove_interface_from_router.assert_called_with(self._router, **{'port_id': self._router.port})
    self.assertIsNone(result)