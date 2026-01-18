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
def test_parsing_auth_v1_1(self):
    data = self.fixtures.load('_v1_1__auth.json')
    data = json.loads(data)
    service_catalog = data['auth']['serviceCatalog']
    catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='1.0')
    entries = catalog.get_entries()
    self.assertEqual(len(entries), 3)
    entry = [e for e in entries if e.service_type == 'cloudFilesCDN'][0]
    self.assertEqual(entry.service_type, 'cloudFilesCDN')
    self.assertIsNone(entry.service_name)
    self.assertEqual(len(entry.endpoints), 2)
    self.assertEqual(entry.endpoints[0].region, 'ORD')
    self.assertEqual(entry.endpoints[0].url, 'https://cdn2.clouddrive.com/v1/MossoCloudFS')
    self.assertEqual(entry.endpoints[0].endpoint_type, 'external')
    self.assertEqual(entry.endpoints[1].region, 'LON')
    self.assertEqual(entry.endpoints[1].endpoint_type, 'external')