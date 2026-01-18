from unittest import mock
import uuid
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from openstack.identity.v3 import registered_limit
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def test_set_model_impl(self):
    enforcer = limit.Enforcer(self._get_usage_for_project)
    self.assertIsInstance(enforcer.model, limit._FlatEnforcer)