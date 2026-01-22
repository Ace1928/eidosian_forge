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
def _metal_v1_devices_1e52437e_bbbb_cccc_dddd_74a9dfd3d3bb_bandwidth(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('node_bandwidth.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])