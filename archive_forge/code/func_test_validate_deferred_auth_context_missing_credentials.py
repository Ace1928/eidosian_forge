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
def test_validate_deferred_auth_context_missing_credentials(self):
    stk = tools.get_stack('test_deferred_auth', self.ctx)
    stk['WebServer'].requires_deferred_auth = True
    cfg.CONF.set_default('deferred_auth_method', 'password')
    ctx = utils.dummy_context(user=None)
    ex = self.assertRaises(exception.MissingCredentialError, self.man._validate_deferred_auth_context, ctx, stk)
    self.assertEqual('Missing required credential: X-Auth-User', str(ex))
    ctx = utils.dummy_context(password=None)
    ex = self.assertRaises(exception.MissingCredentialError, self.man._validate_deferred_auth_context, ctx, stk)
    self.assertEqual('Missing required credential: X-Auth-Key', str(ex))