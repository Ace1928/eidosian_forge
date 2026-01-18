import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_attach_detach_floating_ip(self):
    """Test floating ip create/delete; server add/remove floating ip"""
    if not self.haz_network:
        self.skipTest('No Network service present')

    def _chain_addresses(addresses):
        return itertools.chain(*[*addresses.values()])
    cmd_output = self.server_create()
    name = cmd_output['name']
    self.wait_for_status(name, 'ACTIVE')
    cmd_output = self.openstack('floating ip create ' + 'public', parse_output=True)
    floating_ip = cmd_output.get('floating_ip_address', cmd_output.get('ip', None))
    self.assertNotEqual('', cmd_output['id'])
    self.assertNotEqual('', floating_ip)
    self.addCleanup(self.openstack, 'floating ip delete ' + str(cmd_output['id']))
    raw_output = self.openstack('server add floating ip ' + name + ' ' + floating_ip)
    self.assertEqual('', raw_output)
    wait_time = 0
    while wait_time < 60:
        cmd_output = self.openstack('server show ' + name, parse_output=True)
        if floating_ip not in _chain_addresses(cmd_output['addresses']):
            print('retrying floating IP check')
            wait_time += 10
            time.sleep(10)
        else:
            break
    self.assertIn(floating_ip, _chain_addresses(cmd_output['addresses']))
    raw_output = self.openstack('server remove floating ip ' + name + ' ' + floating_ip)
    self.assertEqual('', raw_output)
    wait_time = 0
    while wait_time < 60:
        cmd_output = self.openstack('server show ' + name, parse_output=True)
        if floating_ip in _chain_addresses(cmd_output['addresses']):
            print('retrying floating IP check')
            wait_time += 10
            time.sleep(10)
        else:
            break
    cmd_output = self.openstack('server show ' + name, parse_output=True)
    self.assertNotIn(floating_ip, _chain_addresses(cmd_output['addresses']))