import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
class TestShowNetworkLog(TestNetworkLog):

    def setUp(self):
        super(TestShowNetworkLog, self).setUp()
        self.neutronclient.show_network_log = mock.Mock(return_value={'log': self.res})
        self.mocked = self.neutronclient.show_network_log
        self.cmd = network_log.ShowNetworkLog(self.app, self.namespace)

    def test_show_filtered_by_id_or_name(self):
        target = self.res['id']
        arglist = [target]
        verifylist = [('network_log', target)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target)
        self.assertEqual(self.ordered_headers, headers)
        self.assertEqual(self.ordered_data, data)