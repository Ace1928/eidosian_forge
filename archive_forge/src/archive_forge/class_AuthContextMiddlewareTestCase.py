import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
class AuthContextMiddlewareTestCase(RestfulTestCase):

    def load_fixtures(self, fixtures):
        self.load_sample_data()
        app_cred_api = PROVIDERS.application_credential_api
        access_rules = [{'id': uuid.uuid4().hex, 'service': self.service['type'], 'method': 'GET', 'path': '/v3/users/*'}]
        app_cred = {'id': uuid.uuid4().hex, 'name': 'appcredtest', 'secret': uuid.uuid4().hex, 'user_id': self.user['id'], 'project_id': self.project['id'], 'description': 'Test Application Credential', 'roles': [{'id': self.role_id}], 'access_rules': access_rules}
        app_cred_ref = app_cred_api.create_application_credential(app_cred)
        self.app_cred_r_id = app_cred_ref['id']
        self.app_cred_r_secret = app_cred_ref['secret']

    def _middleware_request(self, token, extra_environ=None):

        def application(environ, start_response):
            body = b'body'
            headers = [('Content-Type', 'text/html; charset=utf8'), ('Content-Length', str(len(body)))]
            start_response('200 OK', headers)
            return [body]
        app = webtest.TestApp(auth_context.AuthContextMiddleware(application), extra_environ=extra_environ)
        resp = app.get('/', headers={authorization.AUTH_TOKEN_HEADER: token})
        self.assertEqual(b'body', resp.body)
        return resp.request

    def test_auth_context_build_by_middleware(self):
        admin_token = self.get_scoped_token()
        req = self._middleware_request(admin_token)
        self.assertEqual(self.user['id'], req.environ.get(authorization.AUTH_CONTEXT_ENV)['user_id'])

    def test_auth_context_override(self):
        overridden_context = 'OVERRIDDEN_CONTEXT'
        token = uuid.uuid4().hex
        extra_environ = {authorization.AUTH_CONTEXT_ENV: overridden_context}
        req = self._middleware_request(token, extra_environ=extra_environ)
        self.assertEqual(overridden_context, req.environ.get(authorization.AUTH_CONTEXT_ENV))

    def test_unscoped_token_auth_context(self):
        unscoped_token = self.get_unscoped_token()
        req = self._middleware_request(unscoped_token)
        for key in ['project_id', 'domain_id', 'domain_name']:
            self.assertIsNone(req.environ.get(authorization.AUTH_CONTEXT_ENV)[key])

    def test_project_scoped_token_auth_context(self):
        project_scoped_token = self.get_scoped_token()
        req = self._middleware_request(project_scoped_token)
        self.assertEqual(self.project['id'], req.environ.get(authorization.AUTH_CONTEXT_ENV)['project_id'])

    def test_domain_scoped_token_auth_context(self):
        path = '/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id'])
        self.put(path=path)
        domain_scoped_token = self.get_domain_scoped_token()
        req = self._middleware_request(domain_scoped_token)
        self.assertEqual(self.domain['id'], req.environ.get(authorization.AUTH_CONTEXT_ENV)['domain_id'])
        self.assertEqual(self.domain['name'], req.environ.get(authorization.AUTH_CONTEXT_ENV)['domain_name'])

    def test_oslo_context(self):
        token = self.get_scoped_token()
        request_id = uuid.uuid4().hex
        environ = {'openstack.request_id': request_id}
        self._middleware_request(token, extra_environ=environ)
        req_context = oslo_context.context.get_current()
        self.assertEqual(request_id, req_context.request_id)
        self.assertEqual(token, req_context.auth_token)
        self.assertEqual(self.user['id'], req_context.user_id)
        self.assertEqual(self.project['id'], req_context.project_id)
        self.assertIsNone(req_context.domain_id)
        self.assertEqual(self.user['domain_id'], req_context.user_domain_id)
        self.assertEqual(self.project['domain_id'], req_context.project_domain_id)
        self.assertFalse(req_context.is_admin)

    def test_auth_context_app_cred_with_rule(self):

        def application(environ, start_response):
            body = b'body'
            headers = [('Content-Type', 'text/html; charset=utf8'), ('Content-Length', str(len(body)))]
            start_response('200 OK', headers)
            return [body]
        token = self.get_application_credentials_token(self.app_cred_r_id, self.app_cred_r_secret)
        app = webtest.TestApp(auth_context.AuthContextMiddleware(application))
        resp = app.get('/v3/projects/e3a0883d15ff409e98e59d460f583a68', headers={authorization.AUTH_TOKEN_HEADER: token}, status=401)
        self.assertEqual('401 Unauthorized', resp.status)
        app = webtest.TestApp(auth_context.AuthContextMiddleware(application))
        resp = app.get('/v3/users/3879328537914be2b394ddf57a4fc73a', headers={authorization.AUTH_TOKEN_HEADER: token})
        self.assertEqual('200 OK', resp.status)
        self.assertEqual(b'body', resp.body)