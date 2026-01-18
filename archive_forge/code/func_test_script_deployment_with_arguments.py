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
def test_script_deployment_with_arguments(self):
    file_path = '{0}root{0}relative.sh'.format(os.path.sep)
    client = Mock()
    client.put.return_value = file_path
    client.run.return_value = ('', '', 0)
    args = ['arg1', 'arg2', '--option1=test']
    sd = ScriptDeployment(script='echo "foo"', args=args, name=file_path)
    sd.run(self.node, client)
    expected = '%s arg1 arg2 --option1=test' % file_path
    client.run.assert_called_once_with(expected, timeout=None)
    client.reset_mock()
    args = []
    sd = ScriptDeployment(script='echo "foo"', args=args, name=file_path)
    sd.run(self.node, client)
    expected = file_path
    client.run.assert_called_once_with(expected, timeout=None)