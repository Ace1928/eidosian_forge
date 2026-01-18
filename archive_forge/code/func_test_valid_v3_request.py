from unittest import mock
from keystoneauth1 import exceptions as keystone_exc
from oslo_config import cfg
import webob
from heat.common import auth_password
from heat.tests import common
@mock.patch('keystoneauth1.identity.generic.Password')
def test_valid_v3_request(self, mock_password):
    mock_auth = mock.MagicMock()
    mock_password.return_value = mock_auth
    self.patchobject(mock_auth, 'get_access', return_value=FakeAccessInfo(**TOKEN_V3_RESPONSE))
    req = webob.Request.blank('/tenant_id1/')
    req.headers['X_AUTH_USER'] = 'user_name1'
    req.headers['X_AUTH_KEY'] = 'goodpassword'
    req.headers['X_AUTH_URL'] = self.config['auth_uri']
    req.headers['X_USER_DOMAIN_ID'] = 'domain1'
    self.middleware(req.environ, self._start_fake_response)
    mock_password.assert_called_once_with(auth_url=self.config['auth_uri'], password='goodpassword', project_id='tenant_id1', user_domain_id='domain1', username='user_name1')