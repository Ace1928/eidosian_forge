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
def test_stackid_addprefix(self):
    response = self.controller._id_format({'StackName': 'Foo', 'StackId': {u'tenant': u't', u'stack_name': u'Foo', u'stack_id': u'123', u'path': u''}})
    expected = {'StackName': 'Foo', 'StackId': 'arn:openstack:heat::t:stacks/Foo/123'}
    self.assertEqual(expected, response)