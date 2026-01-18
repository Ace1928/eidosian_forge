import time
import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_type_set_unset_project(self):
    name = uuid.uuid4().hex
    cmd_output = self.openstack('volume type create --private ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'volume type delete ' + name)
    self.assertEqual(name, cmd_output['name'])
    raw_output = self.openstack('volume type set --project admin %s' % name)
    self.assertEqual('', raw_output)
    raw_output = self.openstack('volume type unset --project admin %s' % name)
    self.assertEqual('', raw_output)