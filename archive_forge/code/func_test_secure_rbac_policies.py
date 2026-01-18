import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
@ddt.file_data('policy/test_new_acl_personas.yaml')
@ddt.unpack
def test_secure_rbac_policies(self, **kwargs):
    scope = kwargs.get('scope')
    actions = kwargs.get('actions')
    allowed_personas = kwargs.get('allowed', [])
    denied_personas = kwargs.get('denied', [])
    self._test_policy_allowed(scope, actions, allowed_personas)
    self._test_policy_notallowed(scope, actions, denied_personas)