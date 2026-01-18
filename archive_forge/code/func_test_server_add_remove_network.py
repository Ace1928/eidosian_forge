import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_add_remove_network(self):
    name = uuid.uuid4().hex
    cmd_output = self.openstack('server create ' + '--network private ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + '--wait ' + name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(name, cmd_output['name'])
    self.addCleanup(self.openstack, 'server delete --wait ' + name)
    self.openstack('server add network ' + name + ' public')
    wait_time = 0
    while wait_time < 60:
        cmd_output = self.openstack('server show ' + name, parse_output=True)
        if 'public' not in cmd_output['addresses']:
            print('retrying add network check')
            wait_time += 10
            time.sleep(10)
        else:
            break
    addresses = cmd_output['addresses']
    self.assertIn('public', addresses)
    self.openstack('server remove network ' + name + ' public')
    wait_time = 0
    while wait_time < 60:
        cmd_output = self.openstack('server show ' + name, parse_output=True)
        if 'public' in cmd_output['addresses']:
            print('retrying remove network check')
            wait_time += 10
            time.sleep(10)
        else:
            break
    addresses = cmd_output['addresses']
    self.assertNotIn('public', addresses)