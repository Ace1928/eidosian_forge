import json
import os
from unittest import mock
from oslo_config import fixture as config_fixture
from heat.api.aws import exception
import heat.api.cfn.v1.stacks as stacks
from heat.common import exception as heat_exception
from heat.common import identifier
from heat.common import policy
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def _stub_rpc_create_stack_call_success(self, stack_name, engine_parms, engine_args, parameters):
    dummy_req = self._dummy_GET_request(parameters)
    self._stub_enforce(dummy_req, 'CreateStack')
    engine_resp = {u'tenant': u't', u'stack_name': u'wordpress', u'stack_id': u'1', u'path': u''}
    self.m_call.return_value = engine_resp
    return dummy_req