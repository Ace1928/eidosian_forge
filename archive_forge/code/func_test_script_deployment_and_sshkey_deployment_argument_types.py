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
def test_script_deployment_and_sshkey_deployment_argument_types(self):

    class FileObject:

        def __init__(self, name):
            self.name = name

        def read(self):
            return 'bar'
    ScriptDeployment(script='foobar')
    ScriptDeployment(script=u('foobar'))
    ScriptDeployment(script=FileObject('test'))
    SSHKeyDeployment(key='foobar')
    SSHKeyDeployment(key=u('foobar'))
    SSHKeyDeployment(key=FileObject('test'))
    try:
        ScriptDeployment(script=[])
    except TypeError:
        pass
    else:
        self.fail('TypeError was not thrown')
    try:
        SSHKeyDeployment(key={})
    except TypeError:
        pass
    else:
        self.fail('TypeError was not thrown')