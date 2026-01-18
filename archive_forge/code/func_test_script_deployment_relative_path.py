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
def test_script_deployment_relative_path(self):
    client = Mock()
    client.put.return_value = FILE_PATH
    client.run.return_value = ('', '', 0)
    sd = ScriptDeployment(script='echo "foo"', name='relative.sh')
    sd.run(self.node, client)
    client.run.assert_called_once_with(FILE_PATH, timeout=None)