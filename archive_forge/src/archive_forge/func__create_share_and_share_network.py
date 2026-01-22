import ast
import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _create_share_and_share_network(self):
    name = data_utils.rand_name('autotest_share_name')
    description = data_utils.rand_name('autotest_share_description')
    common_share_network = self.client.get_share_network(self.client.share_network)
    share_net_info = utils.get_default_subnet(self.client, common_share_network['id'])
    neutron_net_id = share_net_info['neutron_net_id'] if 'none' not in share_net_info['neutron_net_id'].lower() else None
    neutron_subnet_id = share_net_info['neutron_subnet_id'] if 'none' not in share_net_info['neutron_subnet_id'].lower() else None
    share_network = self.client.create_share_network(neutron_net_id=neutron_net_id, neutron_subnet_id=neutron_subnet_id)
    share_type = self.create_share_type(data_utils.rand_name('test_share_type'), driver_handles_share_servers=True)
    share = self.create_share(share_protocol=self.protocol, size=1, name=name, description=description, share_type=share_type['ID'], share_network=share_network['id'], client=self.client, wait_for_creation=True)
    share = self.client.get_share(share['id'])
    return (share, share_network)