import sys
import json
import unittest
import libcloud.compute.drivers.equinixmetal
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, KeyPair
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.equinixmetal import EquinixMetalNodeDriver
def _metal_v1_ips_01c184f5_1413_4b0b_9f6d_ac993f6c9241(self, method, url, body, headers):
    body = self.fixtures.load('ip_address.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])