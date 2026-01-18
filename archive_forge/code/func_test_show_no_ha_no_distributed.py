from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_show_no_ha_no_distributed(self):
    _router = network_fakes.FakeRouter.create_one_router({'ha': None, 'distributed': None})
    arglist = [_router.name]
    verifylist = [('router', _router.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch.object(self.network_client, 'find_router', return_value=_router):
        columns, data = self.cmd.take_action(parsed_args)
    self.assertNotIn('is_distributed', columns)
    self.assertNotIn('is_ha', columns)