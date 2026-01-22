from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_pool
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestFloatingIPPoolNetwork(network_fakes.TestNetworkV2):

    def setUp(self):
        super(TestFloatingIPPoolNetwork, self).setUp()