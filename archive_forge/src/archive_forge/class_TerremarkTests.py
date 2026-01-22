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
class TerremarkTests(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        VCloudNodeDriver.connectionCls.host = 'test'
        VCloudNodeDriver.connectionCls.conn_class = TerremarkMockHttp
        TerremarkMockHttp.type = None
        self.driver = TerremarkDriver(*VCLOUD_PARAMS)

    def test_list_images(self):
        ret = self.driver.list_images()
        self.assertEqual(ret[0].id, 'https://services.vcloudexpress.terremark.com/api/v0.8/vAppTemplate/5')

    def test_list_sizes(self):
        ret = self.driver.list_sizes()
        self.assertEqual(ret[0].ram, 512)

    def test_create_node(self):
        image = self.driver.list_images()[0]
        size = self.driver.list_sizes()[0]
        node = self.driver.create_node(name='testerpart2', image=image, size=size, ex_vdc='https://services.vcloudexpress.terremark.com/api/v0.8/vdc/224', ex_network='https://services.vcloudexpress.terremark.com/api/v0.8/network/725', ex_cpus=2)
        self.assertTrue(isinstance(node, Node))
        self.assertEqual(node.id, 'https://services.vcloudexpress.terremark.com/api/v0.8/vapp/14031')
        self.assertEqual(node.name, 'testerpart2')

    def test_list_nodes(self):
        ret = self.driver.list_nodes()
        node = ret[0]
        self.assertEqual(node.id, 'https://services.vcloudexpress.terremark.com/api/v0.8/vapp/14031')
        self.assertEqual(node.name, 'testerpart2')
        self.assertEqual(node.state, NodeState.RUNNING)
        self.assertEqual(node.public_ips, [])
        self.assertEqual(node.private_ips, ['10.112.78.69'])

    def test_reboot_node(self):
        node = self.driver.list_nodes()[0]
        ret = self.driver.reboot_node(node)
        self.assertTrue(ret)

    def test_destroy_node(self):
        node = self.driver.list_nodes()[0]
        ret = self.driver.destroy_node(node)
        self.assertTrue(ret)