import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def _doesnt_exist_DescribeKeyPairs(self, method, url, body, headers):
    body = self.fixtures.load('describe_key_pairs_doesnt_exist.xml')
    return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])