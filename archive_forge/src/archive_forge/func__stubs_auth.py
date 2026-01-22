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
def _stubs_auth(self, method='token', trust_scoped=True, user_id=None, auth_ref=None, client=True, project_id=None, stub_trust_context=False, version=3, stub_admin_auth=False):
    self.version = version
    mock_auth_ref = mock.Mock()
    mock_ks_auth = mock.Mock()
    self.method = method
    self.project_id = project_id
    self.client = client
    self.stub_admin_auth = stub_admin_auth
    if method == 'token':
        self.m_token.return_value = mock_ks_auth
    elif method == 'auth_ref':
        self.m_access.return_value = mock_ks_auth
    elif method == 'password':
        ks_auth.Password.return_value = mock_ks_auth
    elif method == 'trust':
        mock_auth_ref.user_id = user_id or 'trustor_user_id'
        mock_auth_ref.project_id = project_id or 'test_tenant_id'
        mock_auth_ref.trust_scoped = trust_scoped
        mock_auth_ref.auth_token = 'atrusttoken'
        self.m_load_auth.return_value = mock_ks_auth
    if client:
        if stub_trust_context:
            mock_ks_auth.get_user_id.return_value = user_id
            mock_ks_auth.get_project_id.return_value = project_id
        mock_ks_auth.get_access.return_value = mock_auth_ref
    if not stub_admin_auth:
        self.m_load_auth.return_value = mock_ks_auth
    else:
        self.mock_admin_ks_auth = mock.Mock()
        self.mock_admin_ks_auth.get_user_id.return_value = '1234'
        self.m_load_auth.return_value = self.mock_admin_ks_auth
    return (mock_ks_auth, mock_auth_ref)