import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.maxihost import MaxihostNodeDriver
def _account_keys(self, method, url, body, headers):
    body = self.fixtures.load('keys.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])