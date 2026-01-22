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
def _metal_v1_projects_4b653fce_6405_4300_9f7d_c587b7888fe5_ips(self, method, url, body, headers):
    body = self.fixtures.load('project_ips.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])