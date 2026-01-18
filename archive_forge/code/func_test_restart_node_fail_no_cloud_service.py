import os
import sys
import libcloud.security
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeAuthPassword
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
def test_restart_node_fail_no_cloud_service(self):
    node = Node(id='dc03', name='dc03', state=NodeState.RUNNING, public_ips=[], private_ips=[], driver=self.driver)
    with self.assertRaises(LibcloudError):
        self.driver.reboot_node(node=node, ex_cloud_service_name='dcoddkinztest03', ex_deployment_slot='Production')