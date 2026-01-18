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
def test_stack_create_total_resources_exceeds_max(self):
    stack_name = 'stack_create_total_resources_exceeds_max'
    params = {}
    tpl = {'heat_template_version': '2014-10-16', 'resources': {'A': {'type': 'GenericResourceType'}, 'B': {'type': 'GenericResourceType'}, 'C': {'type': 'GenericResourceType'}}}
    cfg.CONF.set_override('max_resources_per_stack', 2)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.create_stack, self.ctx, stack_name, tpl, params, None, {})
    self.assertEqual(exception.RequestLimitExceeded, ex.exc_info[0])
    self.assertIn(exception.StackResourceLimitExceeded.msg_fmt, str(ex.exc_info[1]))