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
def _metal_v1_storage_attachments_2c16a96f_bb4f_471b_8e2e_b5820b9e1603(self, method, url, body, headers):
    if method == 'DELETE':
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])