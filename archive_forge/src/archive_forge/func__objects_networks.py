import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeSize
from libcloud.test.secrets import GRIDSCALE_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gridscale import GridscaleNodeDriver
def _objects_networks(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('ex_list_networks.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    else:
        raise ValueError('Invalid method')