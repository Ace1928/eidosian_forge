from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_pool
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestListFloatingIPPoolNetwork(TestFloatingIPPoolNetwork):

    def setUp(self):
        super(TestListFloatingIPPoolNetwork, self).setUp()
        self.cmd = floating_ip_pool.ListFloatingIPPool(self.app, self.namespace)

    def test_floating_ip_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)