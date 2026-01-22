import sys
import random
import string
import unittest
from libcloud.utils.py3 import httplib
from libcloud.common.gandi import GandiException
from libcloud.test.secrets import GANDI_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gandi import GandiNodeDriver
from libcloud.test.common.test_gandi import BaseGandiMockHttp
def _xmlrpc__hosting_ssh_create(self, method, url, body, headers):
    body = self.fixtures.load('ssh_info.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])