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
def _stub_enforce(self, req, action, allowed=True):
    mock_enforce = self.patchobject(policy.Enforcer, 'enforce')
    if allowed:
        mock_enforce.return_value = True
    else:
        mock_enforce.side_effect = heat_exception.Forbidden