import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def _drives_standard_info(self, method, url, body, headers):
    body = self.fixtures.load('drives_standard_info.txt')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])