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
def test_usage_callback_must_be_callable(self):
    invalid_callback_types = [uuid.uuid4().hex, 5, 5.1]
    for invalid_callback in invalid_callback_types:
        self.assertRaises(ValueError, limit.Enforcer, invalid_callback)