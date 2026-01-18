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
def test_stack_validate(self):
    stack_name = 'stack_create_test_validate'
    stk = tools.get_stack(stack_name, self.ctx)
    fc = fakes_nova.FakeClient()
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=fc)
    self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id', return_value=744)
    resource = stk['WebServer']
    resource.properties = properties.Properties(resource.properties_schema, {'ImageId': 'CentOS 5.2', 'KeyName': 'test', 'InstanceType': 'm1.large'}, context=self.ctx)
    stk.validate()
    resource.properties = properties.Properties(resource.properties_schema, {'KeyName': 'test', 'InstanceType': 'm1.large'}, context=self.ctx)
    self.assertRaises(exception.StackValidationFailed, stk.validate)