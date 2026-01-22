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
def _metal_v1_projects_4a4bce6b_d2ef_41f8_95cf_0e2f32996440_bgp_sessions(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('bgp_sessions.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])