import random
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_to_empty(self):
    arglist = [self._network.name, '--name', '', '--description', '', '--dns-domain', '']
    verifylist = [('network', self._network.name), ('description', ''), ('name', ''), ('dns_domain', '')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': '', 'description': '', 'dns_domain': ''}
    self.network_client.update_network.assert_called_once_with(self._network, **attrs)
    self.assertIsNone(result)