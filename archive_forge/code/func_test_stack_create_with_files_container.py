from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_service import threadgroup
from swiftclient import exceptions
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine import environment
from heat.engine import properties
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import service
from heat.engine import stack
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_stack_create_with_files_container(self):
    stack_name = 'env_files_test_stack'
    environment_files = ['env_1', 'env_2']
    files_container = 'test_container'
    fake_get_object = (None, "{'resource_registry': {}}")
    fake_get_container = ({'x-container-bytes-used': 100}, [{'name': '/env/test.yaml'}])
    mock_client = mock.Mock()
    mock_client.get_object.return_value = fake_get_object
    mock_client.get_container.return_value = fake_get_container
    self.patchobject(swift.SwiftClientPlugin, '_create', return_value=mock_client)
    self._test_stack_create(stack_name, environment_files=environment_files, files_container=files_container)
    mock_client.get_container.assert_called_with(files_container)
    mock_client.get_object.assert_called_with(files_container, '/env/test.yaml')