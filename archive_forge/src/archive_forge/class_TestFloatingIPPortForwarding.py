from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestFloatingIPPortForwarding(network_fakes.TestNetworkV2):

    def setUp(self):
        super(TestFloatingIPPortForwarding, self).setUp()
        self.floating_ip = network_fakes.FakeFloatingIP.create_one_floating_ip()
        self.port = network_fakes.create_one_port()
        self.project = identity_fakes_v2.FakeProject.create_one_project()
        self.network_client.find_port = mock.Mock(return_value=self.port)