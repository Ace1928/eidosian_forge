import os
from unittest import mock
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_middleware import request_id
from oslo_policy import opts as policy_opts
from oslo_utils import importutils
import webob
from heat.common import context
from heat.common import exception
from heat.tests import common
class RequestContextMiddlewareTest(common.HeatTestCase):
    scenarios = [('empty_headers', dict(environ=None, headers={}, context_dict={'auth_token': None, 'auth_token_info': None, 'auth_url': None, 'aws_creds': None, 'is_admin': False, 'password': None, 'roles': [], 'show_deleted': False, 'tenant': None, 'tenant_id': None, 'trust_id': None, 'trustor_user_id': None, 'user': None, 'user_id': None, 'username': None})), ('username_password', dict(environ=None, headers={'X-Auth-User': 'my_username', 'X-Auth-Key': 'my_password', 'X-Auth-EC2-Creds': '{"ec2Credentials": {}}', 'X-User-Id': '7a87ff18-31c6-45ce-a186-ec7987f488c3', 'X-Auth-Token': 'atoken', 'X-Project-Name': 'my_tenant', 'X-Project-Id': 'db6808c8-62d0-4d92-898c-d644a6af20e9', 'X-Auth-Url': 'http://192.0.2.1:5000/v1', 'X-Roles': 'role1,role2,role3'}, context_dict={'auth_token': 'atoken', 'auth_url': 'http://192.0.2.1:5000/v1', 'aws_creds': None, 'is_admin': False, 'password': 'my_password', 'roles': ['role1', 'role2', 'role3'], 'show_deleted': False, 'tenant': 'my_tenant', 'tenant_id': 'db6808c8-62d0-4d92-898c-d644a6af20e9', 'trust_id': None, 'trustor_user_id': None, 'user': 'my_username', 'user_id': '7a87ff18-31c6-45ce-a186-ec7987f488c3', 'username': 'my_username'})), ('aws_creds', dict(environ=None, headers={'X-Auth-EC2-Creds': '{"ec2Credentials": {}}', 'X-User-Id': '7a87ff18-31c6-45ce-a186-ec7987f488c3', 'X-Auth-Token': 'atoken', 'X-Project-Name': 'my_tenant', 'X-Project-Id': 'db6808c8-62d0-4d92-898c-d644a6af20e9', 'X-Auth-Url': 'http://192.0.2.1:5000/v1', 'X-Roles': 'role1,role2,role3'}, context_dict={'auth_token': 'atoken', 'auth_url': 'http://192.0.2.1:5000/v1', 'aws_creds': '{"ec2Credentials": {}}', 'is_admin': False, 'password': None, 'roles': ['role1', 'role2', 'role3'], 'show_deleted': False, 'tenant': 'my_tenant', 'tenant_id': 'db6808c8-62d0-4d92-898c-d644a6af20e9', 'trust_id': None, 'trustor_user_id': None, 'user': None, 'user_id': '7a87ff18-31c6-45ce-a186-ec7987f488c3', 'username': None})), ('token_creds', dict(environ={'keystone.token_info': {'info': 123}}, headers={'X-User-Id': '7a87ff18-31c6-45ce-a186-ec7987f488c3', 'X-Auth-Token': 'atoken2', 'X-Project-Name': 'my_tenant2', 'X-Project-Id': 'bb9108c8-62d0-4d92-898c-d644a6af20e9', 'X-Auth-Url': 'http://192.0.2.1:5000/v1', 'X-Roles': 'role1,role2,role3'}, context_dict={'auth_token': 'atoken2', 'auth_token_info': {'info': 123}, 'auth_url': 'http://192.0.2.1:5000/v1', 'aws_creds': None, 'is_admin': False, 'password': None, 'roles': ['role1', 'role2', 'role3'], 'show_deleted': False, 'tenant': 'my_tenant2', 'tenant_id': 'bb9108c8-62d0-4d92-898c-d644a6af20e9', 'trust_id': None, 'trustor_user_id': None, 'user': None, 'user_id': '7a87ff18-31c6-45ce-a186-ec7987f488c3', 'username': None}))]

    def setUp(self):
        super(RequestContextMiddlewareTest, self).setUp(mock_find_file=False)
        self.fixture = self.useFixture(config_fixture.Config())
        self.fixture.conf(args=['--config-dir', policy_path])
        policy_opts.set_defaults(cfg.CONF, 'check_admin.json')

    def test_context_middleware(self):
        middleware = context.ContextMiddleware(None, None)
        request = webob.Request.blank('/stacks', headers=self.headers, environ=self.environ)
        self.assertIsNone(middleware.process_request(request))
        ctx = request.context.to_dict()
        for k, v in self.context_dict.items():
            self.assertEqual(v, ctx[k], 'Key %s values do not match' % k)
        self.assertIsNotNone(ctx.get('request_id'))

    def test_context_middleware_with_requestid(self):
        middleware = context.ContextMiddleware(None, None)
        request = webob.Request.blank('/stacks', headers=self.headers, environ=self.environ)
        req_id = 'req-5a63f0d7-1b69-447b-b621-4ea87cc7186d'
        request.environ[request_id.ENV_REQUEST_ID] = req_id
        self.assertIsNone(middleware.process_request(request))
        ctx = request.context.to_dict()
        for k, v in self.context_dict.items():
            self.assertEqual(v, ctx[k], 'Key %s values do not match' % k)
        self.assertEqual(ctx.get('request_id'), req_id, 'Key request_id values do not match')