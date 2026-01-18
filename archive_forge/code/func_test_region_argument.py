import sys
import unittest
from unittest.mock import patch
from libcloud.http import LibcloudConnection
from libcloud.test import no_internet
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import OVH_PARAMS
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ovh import OvhNodeDriver
from libcloud.test.common.test_ovh import BaseOvhMockHttp
def test_region_argument(self):
    driver = OvhNodeDriver(*OVH_PARAMS)
    self.assertEqual(driver.connection.host, 'api.ovh.com')
    driver = OvhNodeDriver(*OVH_PARAMS, region=None)
    self.assertEqual(driver.connection.host, 'api.ovh.com')
    driver = OvhNodeDriver(*OVH_PARAMS, region='ca')
    driver = OvhNodeDriver(*OVH_PARAMS, region='eu')
    self.assertEqual(driver.connection.host, 'eu.api.ovh.com')
    driver = OvhNodeDriver(*OVH_PARAMS, region='ca')
    self.assertEqual(driver.connection.host, 'ca.api.ovh.com')