import json
from unittest import mock
from oslo_config import cfg
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.stacks as stacks
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_show_without_resolve_outputs(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'show', True)
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '6')
    req = self._get('/stacks/%(stack_name)s/%(stack_id)s' % identity, params={'resolve_outputs': False})
    parameters = {u'DBUsername': u'admin', u'LinuxDistribution': u'F17', u'InstanceType': u'm1.large', u'DBRootPassword': u'admin', u'DBPassword': u'admin', u'DBName': u'wordpress'}
    engine_resp = [{u'stack_identity': dict(identity), u'updated_time': u'2012-07-09T09:13:11Z', u'parameters': parameters, u'stack_status_reason': u'Stack successfully created', u'creation_time': u'2012-07-09T09:12:45Z', u'stack_name': identity.stack_name, u'notification_topics': [], u'stack_action': u'CREATE', u'stack_status': u'COMPLETE', u'description': u'blah', u'disable_rollback': True, u'timeout_mins': 60, u'capabilities': []}]
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=engine_resp)
    response = self.controller.show(req, tenant_id=identity.tenant, stack_name=identity.stack_name, stack_id=identity.stack_id)
    expected = {'stack': {'links': [{'href': self._url(identity), 'rel': 'self'}], 'id': '6', u'updated_time': u'2012-07-09T09:13:11Z', u'parameters': parameters, u'description': u'blah', u'stack_status_reason': u'Stack successfully created', u'creation_time': u'2012-07-09T09:12:45Z', u'stack_name': identity.stack_name, u'stack_status': u'CREATE_COMPLETE', u'capabilities': [], u'notification_topics': [], u'disable_rollback': True, u'timeout_mins': 60}}
    self.assertEqual(expected, response)
    mock_call.assert_called_once_with(req.context, ('show_stack', {'stack_identity': dict(identity), 'resolve_outputs': False}), version='1.20')