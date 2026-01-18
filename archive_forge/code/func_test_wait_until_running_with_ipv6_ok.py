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
def test_wait_until_running_with_ipv6_ok(self):
    RackspaceMockHttp.type = 'IPV6'
    node2, ips = self.driver.wait_until_running(nodes=[self.node], wait_period=1, force_ipv4=False, timeout=0.5)[0]
    self.assertEqual(self.node.uuid, node2.uuid)
    self.assertEqual(['2001:DB8::1'], ips)