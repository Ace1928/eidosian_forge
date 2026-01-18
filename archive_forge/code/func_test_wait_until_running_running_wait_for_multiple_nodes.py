import os
import sys
import time
import unittest
from unittest.mock import Mock, patch
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.utils.py3 import u, httplib, assertRaisesRegex
from libcloud.compute.ssh import BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.compute.base import Node, NodeAuthPassword
from libcloud.test.secrets import RACKSPACE_PARAMS
from libcloud.compute.types import NodeState, LibcloudError, DeploymentError
from libcloud.compute.deployment import (
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.rackspace import RackspaceFirstGenNodeDriver as Rackspace
def test_wait_until_running_running_wait_for_multiple_nodes(self):
    RackspaceMockHttp.type = 'MULTIPLE_NODES'
    nodes = self.driver.wait_until_running(nodes=[self.node, self.node2], wait_period=0.1, timeout=0.5)
    self.assertEqual(self.node.uuid, nodes[0][0].uuid)
    self.assertEqual(self.node2.uuid, nodes[1][0].uuid)
    self.assertEqual(['67.23.21.33'], nodes[0][1])
    self.assertEqual(['67.23.21.34'], nodes[1][1])