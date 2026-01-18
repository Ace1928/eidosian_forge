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
def test_validate_err_denied_policy(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'validate_template', False)
    template = {u'Foo': u'bar'}
    body = {'template': template}
    req = self._post('/validate', json.dumps(body))
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.validate_template, req, tenant_id=self.tenant, body=body)
    self.assertEqual(403, resp.status_int)
    self.assertIn('403 Forbidden', str(resp))