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
def test_username_length(self):
    """Test that user names >255 characters are properly truncated."""
    self._stubs_auth()
    ctx = utils.dummy_context()
    ctx.trust_id = None
    long_user_name = 'U' * 255 + 'S'
    good_user_name = 'U' * 254 + 'S'
    mock_user = mock.Mock()
    mock_user.id = 'auser123'
    self.mock_ks_v3_client.users.create.return_value = mock_user
    self.mock_ks_v3_client.roles.list.return_value = self._mock_roles_list()
    self.mock_ks_v3_client.roles.grant.return_value = None
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    heat_ks_client.create_stack_user(long_user_name, password='password')
    self.mock_ks_v3_client.users.create.assert_called_once_with(name=good_user_name, password='password', default_project=ctx.tenant_id)
    self.mock_ks_v3_client.roles.list.assert_called_once_with(name='heat_stack_user')
    self.mock_ks_v3_client.roles.grant.assert_called_once_with(project=ctx.tenant_id, role='4546', user='auser123')
    self._validate_stub_auth()