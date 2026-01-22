import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
class BaseEC2Tests(LibcloudTestCase):

    def test_instantiate_driver_valid_regions(self):
        regions = VALID_EC2_REGIONS
        regions = [d for d in regions if d != 'nimbus' and d != 'cn-north-1']
        region_endpoints = [EC2NodeDriver(*EC2_PARAMS, **{'region': region}).connection.host for region in regions]
        self.assertEqual(len(region_endpoints), len(set(region_endpoints)), 'Multiple Region Drivers were given the same API endpoint')

    def test_instantiate_driver_invalid_regions(self):
        for region in ['invalid', 'nimbus']:
            try:
                EC2NodeDriver(*EC2_PARAMS, **{'region': region})
            except ValueError:
                pass
            else:
                self.fail('Invalid region, but exception was not thrown')

    def test_list_sizes_valid_regions(self):
        unsupported_regions = list()
        for region in VALID_EC2_REGIONS:
            no_pricing = region in ['cn-north-1']
            driver = EC2NodeDriver(*EC2_PARAMS, **{'region': region})
            try:
                sizes = driver.list_sizes()
                if no_pricing:
                    self.assertTrue(all([s.price is None for s in sizes]))
            except Exception:
                unsupported_regions.append(region)
        if unsupported_regions:
            self.fail('Cannot list sizes from ec2 regions: %s' % unsupported_regions)