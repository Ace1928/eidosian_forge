import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_region_attributes(self):
    self.assertIsNone(self.driver._region_dict)
    self.assertIsNone(self.driver._region_list)
    regions = self.driver.ex_list_regions()
    self.assertEqual(len(self.driver.region_list), len(regions))
    self.assertEqual(len(self.driver.region_dict), len(regions))
    for region, fetched_region in zip(self.driver.region_list, regions):
        self.assertEqual(region.id, fetched_region.id)
        self.assertEqual(region.name, fetched_region.name)
        self.assertEqual(region.status, fetched_region.status)