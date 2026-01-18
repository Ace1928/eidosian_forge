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
def test_zone_attributes(self):
    self.assertIsNone(self.driver._zone_dict)
    self.assertIsNone(self.driver._zone_list)
    zones = self.driver.ex_list_zones()
    self.assertEqual(len(self.driver.zone_list), len(zones))
    self.assertEqual(len(self.driver.zone_dict), len(zones))
    for zone, fetched_zone in zip(self.driver.zone_list, zones):
        self.assertEqual(zone.id, fetched_zone.id)
        self.assertEqual(zone.name, fetched_zone.name)
        self.assertEqual(zone.status, fetched_zone.status)