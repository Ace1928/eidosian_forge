import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_create_with_security_group(self):
    """Test server create with security group ID and name"""
    if not self.haz_network:
        self.skipTest('No Network service present')
    sg_name1 = uuid.uuid4().hex
    security_group1 = self.openstack('security group create ' + sg_name1, parse_output=True)
    self.addCleanup(self.openstack, 'security group delete ' + sg_name1)
    sg_name2 = uuid.uuid4().hex
    security_group2 = self.openstack('security group create ' + sg_name2, parse_output=True)
    self.addCleanup(self.openstack, 'security group delete ' + sg_name2)
    server_name = uuid.uuid4().hex
    server = self.openstack('server create ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + '--security-group ' + str(security_group1['id']) + ' ' + '--security-group ' + security_group2['name'] + ' ' + self.network_arg + ' ' + server_name, parse_output=True)
    self.addCleanup(self.openstack, 'server delete --wait ' + server_name)
    self.assertIsNotNone(server['id'])
    self.assertEqual(server_name, server['name'])
    sec_grp = ''
    for sec in server['security_groups']:
        sec_grp += sec['name']
    self.assertIn(str(security_group1['id']), sec_grp)
    self.assertIn(str(security_group2['id']), sec_grp)
    self.wait_for_status(server_name, 'ACTIVE')
    server = self.openstack('server show ' + server_name, parse_output=True)
    sec_grp = ''
    for sec in server['security_groups']:
        sec_grp += sec['name']
    self.assertIn(sg_name1, sec_grp)
    self.assertIn(sg_name2, sec_grp)