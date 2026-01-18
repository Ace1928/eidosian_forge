import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_add_remove_port(self):
    name = uuid.uuid4().hex
    cmd_output = self.openstack('server create ' + '--network private ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + '--wait ' + name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(name, cmd_output['name'])
    self.addCleanup(self.openstack, 'server delete --wait ' + name)
    port_name = uuid.uuid4().hex
    cmd_output = self.openstack('port list', parse_output=True)
    self.assertNotIn(port_name, cmd_output)
    cmd_output = self.openstack('port create ' + '--network private ' + port_name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    ip_address = cmd_output['fixed_ips'][0]['ip_address']
    self.addCleanup(self.openstack, 'port delete ' + port_name)
    self.openstack('server add port ' + name + ' ' + port_name)
    wait_time = 0
    while wait_time < 60:
        cmd_output = self.openstack('server show ' + name, parse_output=True)
        if ip_address not in cmd_output['addresses']['private']:
            print('retrying add port check')
            wait_time += 10
            time.sleep(10)
        else:
            break
    addresses = cmd_output['addresses']['private']
    self.assertIn(ip_address, addresses)
    self.openstack('server remove port ' + name + ' ' + port_name)
    wait_time = 0
    while wait_time < 60:
        cmd_output = self.openstack('server show ' + name, parse_output=True)
        if ip_address in cmd_output['addresses']['private']:
            print('retrying add port check')
            wait_time += 10
            time.sleep(10)
        else:
            break
    addresses = cmd_output['addresses']['private']
    self.assertNotIn(ip_address, addresses)