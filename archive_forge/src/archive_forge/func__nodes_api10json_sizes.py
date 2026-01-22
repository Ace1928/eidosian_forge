import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VPSNET_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vpsnet import VPSNetNodeDriver
def _nodes_api10json_sizes(self, method, url, body, headers):
    body = '[{"slice":{"virtual_machine_id":8592,"id":12256,"consumer_id":0}},\n                   {"slice":{"virtual_machine_id":null,"id":12258,"consumer_id":0}},\n                   {"slice":{"virtual_machine_id":null,"id":12434,"consumer_id":0}}]'
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])