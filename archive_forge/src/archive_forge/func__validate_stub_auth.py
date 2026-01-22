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
def _validate_stub_auth(self):
    if self.method == 'token':
        self.m_token.assert_called_once_with(token='abcd1234', endpoint='http://server.test:5000/v3')
    else:
        self.m_token.assert_not_called()
    if self.method == 'auth_ref':
        if self.version == 3:
            access_type = ks_access.AccessInfoV3
        else:
            access_type = ks_access.AccessInfoV2
        self.m_access.assert_called_once_with(auth_ref=utils.AnyInstance(access_type), auth_url='http://server.test:5000/v3')
    else:
        self.m_access.assert_not_called()
    if self.method == 'password':
        self.m_password.assert_called_once_with(auth_url='http://server.test:5000/v3', username='test_username', password='password', project_id=self.project_id or 'test_tenant_id', user_domain_id='adomain123')
    else:
        self.m_password.assert_not_called()
    if self.method == 'trust':
        self.m_load_auth.assert_called_once_with(cfg.CONF, 'trustee', trust_id='atrust123')
    else:
        self.m_load_auth.assert_not_called()
    if self.client:
        self.m_client.assert_any_call(session=utils.AnyInstance(ks_session.Session), connect_retries=2, interface='publicURL', region_name=None)
    if self.stub_admin_auth:
        self.mock_admin_ks_auth.get_user_id.assert_called_once_with(utils.AnyInstance(ks_session.Session))