import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import FileFixtures
from libcloud.common.digitalocean import DigitalOceanBaseDriver
def _v2_actions(self, method, url, body, headers):
    body = self.fixtures.load('_v2_actions.json')
    return (self.response[self.type], body, {}, httplib.responses[self.response[self.type]])