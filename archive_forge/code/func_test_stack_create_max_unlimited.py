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
@mock.patch.object(instances.Instance, 'validate')
@mock.patch.object(stack.Stack, 'total_resources')
def test_stack_create_max_unlimited(self, total_res_mock, validate_mock):
    total_res_mock.return_value = 9999
    validate_mock.return_value = None
    cfg.CONF.set_override('max_resources_per_stack', -1)
    stack_name = 'service_create_test_max_unlimited'
    self._test_stack_create(stack_name)