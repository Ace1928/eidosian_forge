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
def test_wait_until_running_timeout(self):
    RackspaceMockHttp.type = 'TIMEOUT'
    try:
        self.driver.wait_until_running(nodes=[self.node], wait_period=0.1, timeout=0.2)
    except LibcloudError as e:
        self.assertTrue(e.value.find('Timed out') != -1)
    else:
        self.fail('Exception was not thrown')