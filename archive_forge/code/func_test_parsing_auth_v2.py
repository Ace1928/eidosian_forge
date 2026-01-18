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
def test_parsing_auth_v2(self):
    data = self.fixtures.load('_v2_0__auth.json')
    data = json.loads(data)
    service_catalog = data['access']['serviceCatalog']
    catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='2.0')
    entries = catalog.get_entries()
    self.assertEqual(len(entries), 10)
    entry = [e for e in entries if e.service_name == 'cloudServers'][0]
    self.assertEqual(entry.service_type, 'compute')
    self.assertEqual(entry.service_name, 'cloudServers')
    self.assertEqual(len(entry.endpoints), 1)
    self.assertIsNone(entry.endpoints[0].region)
    self.assertEqual(entry.endpoints[0].url, 'https://servers.api.rackspacecloud.com/v1.0/1337')
    self.assertEqual(entry.endpoints[0].endpoint_type, 'external')