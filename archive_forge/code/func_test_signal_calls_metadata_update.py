from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import identifier
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import dependencies
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as ins
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch.object(heat_plugin.HeatClientPlugin, 'get_heat_cfn_url')
@mock.patch.object(res.Resource, 'metadata_update')
@mock.patch.object(res.Resource, 'signal')
@mock.patch.object(service.EngineService, '_get_stack')
def test_signal_calls_metadata_update(self, mock_get, mock_signal, mock_update, mock_get_cfn):
    mock_get_cfn.return_value = 'http://server.test:8000/v1'
    self.patchobject(keystone.KeystoneClientPlugin, '_create', return_value=fake_ks.FakeKeystoneClient())
    stk = tools.get_stack('signal_reception', self.ctx, policy_template)
    self.stack = stk
    stk.store()
    stk.create()
    s = stack_object.Stack.get_by_id(self.ctx, self.stack.id)
    mock_get.return_value = s
    mock_signal.return_value = True
    mock_update.return_value = None
    self.eng.resource_signal(self.ctx, dict(self.stack.identifier()), 'WebServerScaleDownPolicy', None, sync_call=True)
    mock_get.assert_called_once_with(self.ctx, self.stack.identifier())
    mock_signal.assert_called_once_with(mock.ANY, False)
    mock_update.assert_called_once_with()