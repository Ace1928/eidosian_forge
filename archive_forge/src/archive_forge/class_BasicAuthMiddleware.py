import base64
import binascii
import logging
import bcrypt
import webob
from oslo_config import cfg
from oslo_middleware import base
class BasicAuthMiddleware(base.ConfigurableMiddleware):
    """Middleware which performs HTTP basic authentication on requests"""

    def __init__(self, application, conf=None):
        super().__init__(application, conf)
        self.auth_file = cfg.CONF.oslo_middleware.http_basic_auth_user_file
        validate_auth_file(self.auth_file)

    def format_exception(self, e):
        result = {'error': {'message': str(e), 'code': 401}}
        headers = [('Content-Type', 'application/json')]
        return webob.Response(content_type='application/json', status_code=401, json_body=result, headerlist=headers)

    @webob.dec.wsgify
    def __call__(self, req):
        try:
            token = parse_header(req.environ)
            username, password = parse_token(token)
            req.environ.update(authenticate(self.auth_file, username, password))
            return self.application
        except Exception as e:
            response = self.format_exception(e)
            return self.process_response(response)