import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
def test_parsing_auth_v3(self):
    data = self.fixtures.load('_v3__auth.json')
    data = json.loads(data)
    service_catalog = data['token']['catalog']
    catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='3.x')
    entries = catalog.get_entries()
    self.assertEqual(len(entries), 6)
    entry = [e for e in entries if e.service_type == 'volume'][0]
    self.assertEqual(entry.service_type, 'volume')
    self.assertIsNone(entry.service_name)
    self.assertEqual(len(entry.endpoints), 3)
    self.assertEqual(entry.endpoints[0].region, 'regionOne')
    self.assertEqual(entry.endpoints[0].endpoint_type, 'external')
    self.assertEqual(entry.endpoints[1].region, 'regionOne')
    self.assertEqual(entry.endpoints[1].endpoint_type, 'admin')
    self.assertEqual(entry.endpoints[2].region, 'regionOne')
    self.assertEqual(entry.endpoints[2].endpoint_type, 'internal')