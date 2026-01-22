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
class OutscaleFCUTests(LibcloudTestCase):

    def setUp(self):
        OutscaleSASNodeDriver.connectionCls.conn_class = FCUMockHttp
        EC2MockHttp.use_param = 'Action'
        EC2MockHttp.type = None
        self.driver = OutscaleSASNodeDriver(key=EC2_PARAMS[0], secret=EC2_PARAMS[1], host='some.fcucloud.com')

    def test_ex_describe_quotas(self):
        is_truncated, quota = self.driver.ex_describe_quotas()
        self.assertTrue(is_truncated == 'true')
        self.assertTrue('global' in quota.keys())
        self.assertTrue('vpc-00000000' in quota.keys())

    def test_ex_describe_product_types(self):
        product_types = self.driver.ex_describe_product_types()
        pt = {}
        for e in product_types:
            pt[e['productTypeId']] = e['description']
        self.assertTrue('0001' in pt.keys())
        self.assertTrue('MapR' in pt.values())
        self.assertTrue(pt['0002'] == 'Windows')

    def test_ex_describe_instance_instance_types(self):
        instance_types = self.driver.ex_describe_instance_types()
        it = {}
        for e in instance_types:
            it[e['name']] = e['memory']
        self.assertTrue('og4.4xlarge' in it.keys())
        self.assertTrue('oc2.8xlarge' in it.keys())
        self.assertTrue('68718428160' in it.values())
        self.assertTrue(it['m3.large'] == '8050966528')

    def test_ex_get_product_type(self):
        product_type = self.driver.ex_get_product_type('ami-29ab9e54')
        self.assertTrue(product_type['productTypeId'] == '0002')
        self.assertTrue(product_type['description'] == 'Windows')

    def test_ex_modify_instance_keypair(self):
        r = self.driver.ex_modify_instance_keypair('i-57292bc5', 'key_name')
        self.assertTrue(r)