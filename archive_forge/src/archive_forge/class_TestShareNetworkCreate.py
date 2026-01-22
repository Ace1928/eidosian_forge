from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareNetworkCreate(TestShareNetwork):

    def setUp(self):
        super(TestShareNetworkCreate, self).setUp()
        self.share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.create.return_value = self.share_network
        self.cmd = osc_share_networks.CreateShareNetwork(self.app, None)
        self.data = self.share_network._info.values()
        self.columns = self.share_network._info.keys()

    @ddt.data('table', 'yaml')
    def test_share_network_create_formatter(self, formatter):
        arglist = ['-f', formatter]
        verifylist = [('formatter', formatter)]
        expected_data = self.share_network._info
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.share_networks_mock.create.assert_called_once_with(name=None, description=None, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(expected_data.values(), data)

    def test_share_network_create_with_args(self):
        fake_neutron_net_id = str(uuid.uuid4())
        fake_neutron_subnet_id = str(uuid.uuid4())
        fake_az = mock.Mock()
        fake_az.name = 'nova'
        fake_az.id = str(uuid.uuid4())
        arglist = ['--name', 'zorilla-net', '--description', 'fastest-backdoor-network-ever', '--neutron-net-id', fake_neutron_net_id, '--neutron-subnet-id', fake_neutron_subnet_id, '--availability-zone', 'nova']
        verifylist = [('name', 'zorilla-net'), ('description', 'fastest-backdoor-network-ever'), ('neutron_net_id', fake_neutron_net_id), ('neutron_subnet_id', fake_neutron_subnet_id), ('availability_zone', 'nova')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.find_resource', return_value=fake_az):
            columns, data = self.cmd.take_action(parsed_args)
        self.share_networks_mock.create.assert_called_once_with(name='zorilla-net', description='fastest-backdoor-network-ever', neutron_net_id=fake_neutron_net_id, neutron_subnet_id=fake_neutron_subnet_id, availability_zone='nova')
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)