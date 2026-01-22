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
def _metal_v1_projects_3d27fd13_0466_4878_be22_9a4b5595a3df_plans(self, method, url, body, headers):
    body = self.fixtures.load('plans.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])