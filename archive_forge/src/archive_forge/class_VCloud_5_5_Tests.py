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
class VCloud_5_5_Tests(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        VCloudNodeDriver.connectionCls.host = 'test'
        VCloudNodeDriver.connectionCls.conn_class = VCloud_5_5_MockHttp
        VCloud_5_5_MockHttp.type = None
        self.driver = VCloudNodeDriver(*VCLOUD_PARAMS, **{'api_version': '5.5'})
        self.assertTrue(isinstance(self.driver, VCloud_5_5_NodeDriver))

    def test_ex_create_snapshot(self):
        node = Node('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6b', 'testNode', NodeState.RUNNING, [], [], self.driver)
        self.driver.ex_create_snapshot(node)

    def test_ex_remove_snapshots(self):
        node = Node('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6b', 'testNode', NodeState.RUNNING, [], [], self.driver)
        self.driver.ex_remove_snapshots(node)

    def test_ex_revert_to_snapshot(self):
        node = Node('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6b', 'testNode', NodeState.RUNNING, [], [], self.driver)
        self.driver.ex_revert_to_snapshot(node)

    def test_ex_acquire_mks_ticket(self):
        node = self.driver.ex_find_node('testNode')
        self.driver.ex_acquire_mks_ticket(node.id)

    def test_get_auth_headers(self):
        headers = self.driver.connection._get_auth_headers()
        self.assertEqual(headers['Accept'], 'application/*+xml;version=5.5')