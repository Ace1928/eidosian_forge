import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
class VCloud_5_1_Tests(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        VCloudNodeDriver.connectionCls.host = 'test'
        VCloudNodeDriver.connectionCls.conn_class = VCloud_1_5_MockHttp
        VCloud_1_5_MockHttp.type = None
        self.driver = VCloudNodeDriver(*VCLOUD_PARAMS, **{'api_version': '5.1'})
        self.assertTrue(isinstance(self.driver, VCloud_5_1_NodeDriver))

    def _test_create_node_valid_ex_vm_memory(self):
        values = [4, 1024, 4096]
        image = self.driver.list_images()[0]
        size = self.driver.list_sizes()[0]
        for value in values:
            self.driver.create_node(name='testerpart2', image=image, size=size, vdc='https://services.vcloudexpress.terremark.com/api/v0.8/vdc/224', network='https://services.vcloudexpress.terremark.com/api/v0.8/network/725', cpus=2, ex_vm_memory=value)

    def test_create_node_invalid_ex_vm_memory(self):
        values = [1, 3, 7]
        image = self.driver.list_images()[0]
        size = self.driver.list_sizes()[0]
        for value in values:
            try:
                self.driver.create_node(name='testerpart2', image=image, size=size, vdc='https://services.vcloudexpress.terremark.com/api/v0.8/vdc/224', network='https://services.vcloudexpress.terremark.com/api/v0.8/network/725', cpus=2, ex_vm_memory=value)
            except ValueError:
                pass
            else:
                self.fail('Exception was not thrown')

    def test_list_images(self):
        ret = self.driver.list_images()
        self.assertEqual('https://vm-vcloud/api/vAppTemplate/vappTemplate-ac1bc027-bf8c-4050-8643-4971f691c158', ret[0].id)