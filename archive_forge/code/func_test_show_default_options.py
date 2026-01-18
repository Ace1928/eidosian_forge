from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_show_default_options(self):
    arglist = [self.ndp_proxy.id]
    verifylist = [('ndp_proxy', self.ndp_proxy.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.find_ndp_proxy.assert_called_once_with(self.ndp_proxy.id, ignore_missing=False)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)