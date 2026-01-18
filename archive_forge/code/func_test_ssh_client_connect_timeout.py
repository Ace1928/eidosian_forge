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
def test_ssh_client_connect_timeout(self):
    mock_ssh_client = Mock()
    mock_ssh_client.connect = Mock()
    mock_ssh_client.connect.side_effect = IOError('bam')
    try:
        self.driver._ssh_client_connect(ssh_client=mock_ssh_client, wait_period=0.1, timeout=0.2)
    except LibcloudError as e:
        self.assertTrue(e.value.find('Giving up') != -1)
    else:
        self.fail('Exception was not thrown')