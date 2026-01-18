from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_router_list_agents(self):
    arglist = ['--agent', self._testagent.id]
    verifylist = [('agent', self._testagent.id)]
    attrs = {self._testagent.id}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.agent_hosted_routers(*attrs)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))