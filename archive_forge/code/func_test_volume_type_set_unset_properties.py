import time
import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_type_set_unset_properties(self):
    name = uuid.uuid4().hex
    cmd_output = self.openstack('volume type create --private ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'volume type delete ' + name)
    self.assertEqual(name, cmd_output['name'])
    raw_output = self.openstack('volume type set --property a=b --property c=d %s' % name)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('volume type show %s' % name, parse_output=True)
    self.assertEqual({'a': 'b', 'c': 'd'}, cmd_output['properties'])
    raw_output = self.openstack('volume type unset --property a %s' % name)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('volume type show %s' % name, parse_output=True)
    self.assertEqual({'c': 'd'}, cmd_output['properties'])