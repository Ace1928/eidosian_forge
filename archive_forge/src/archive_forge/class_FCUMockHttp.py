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
class FCUMockHttp(EC2MockHttp):
    fixtures = ComputeFileFixtures('fcu')

    def _DescribeQuotas(self, method, url, body, headers):
        body = self.fixtures.load('ex_describe_quotas.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeProductTypes(self, method, url, body, headers):
        body = self.fixtures.load('ex_describe_product_types.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeInstanceTypes(self, method, url, body, headers):
        body = self.fixtures.load('ex_describe_instance_types.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GetProductType(self, method, url, body, headers):
        body = self.fixtures.load('ex_get_product_type.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ModifyInstanceKeypair(self, method, url, body, headers):
        body = self.fixtures.load('ex_modify_instance_keypair.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])