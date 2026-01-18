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
def test_serialize_create(self):
    result = {'stack': {'id': '1', 'links': [{'href': 'location', 'rel': 'self'}]}}
    response = webob.Response()
    response = self.serializer.create(response, result)
    self.assertEqual(201, response.status_int)
    self.assertEqual('location', response.headers['Location'])
    self.assertEqual('application/json', response.headers['Content-Type'])