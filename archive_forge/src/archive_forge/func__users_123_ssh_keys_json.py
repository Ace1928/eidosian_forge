import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import ONAPP_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.onapp import OnAppNodeDriver
def _users_123_ssh_keys_json(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('list_key_pairs.json')
    else:
        body = self.fixtures.load('import_key_pair.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])