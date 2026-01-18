import json
from unittest import mock
import uuid
from keystoneauth1 import access as ks_access
from keystoneauth1 import exceptions as kc_exception
from keystoneauth1.identity import access as ks_auth_access
from keystoneauth1.identity import generic as ks_auth
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session as ks_session
from keystoneauth1 import token_endpoint as ks_token_endpoint
from keystoneclient.v3 import client as kc_v3
from keystoneclient.v3 import domains as kc_v3_domains
from oslo_config import cfg
from heat.common import config
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients.os.keystone import heat_keystoneclient
from heat.tests import common
from heat.tests import utils
def test_regenerate_trust_context_with_exist_trust_id(self):
    """Test regenerate_trust_context."""
    self._stubs_auth(method='trust')
    cfg.CONF.set_override('deferred_auth_method', 'trusts')
    ctx = utils.dummy_context()
    ctx.trust_id = 'atrust123'
    ctx.trustor_user_id = 'trustor_user_id'

    class MockTrust(object):
        id = 'dtrust123'
    self.mock_ks_v3_client.trusts.create.return_value = MockTrust()
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    trust_context = heat_ks_client.regenerate_trust_context()
    self.assertEqual('dtrust123', trust_context.trust_id)
    self.mock_ks_v3_client.trusts.delete.assert_called_once_with(ctx.trust_id)