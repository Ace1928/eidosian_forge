import json
import os
from oslo_config import cfg
from oslo_log import log as logging
class NoAuthProtocol(object):

    def __init__(self, app, conf):
        self.conf = conf
        self.app = app
        self._token_info = {}
        response_file = cfg.CONF.noauth.token_response
        if os.path.exists(response_file):
            with open(response_file) as f:
                self._token_info = json.loads(f.read())

    def __call__(self, env, start_response):
        """Handle incoming request.

        Authenticate send downstream on success. Reject request if
        we can't authenticate.
        """
        LOG.debug('Authenticating user token')
        env.update(self._build_user_headers(env))
        return self.app(env, start_response)

    def _build_user_headers(self, env):
        """Build headers that represent authenticated user from auth token."""
        username = env.get('HTTP_X_AUTH_USER', 'admin')
        project = env.get('HTTP_X_AUTH_PROJECT', 'admin')
        headers = {'HTTP_X_IDENTITY_STATUS': 'Confirmed', 'HTTP_X_PROJECT_ID': project, 'HTTP_X_PROJECT_NAME': project, 'HTTP_X_USER_ID': username, 'HTTP_X_USER_NAME': username, 'HTTP_X_ROLES': 'admin', 'HTTP_X_SERVICE_CATALOG': {}, 'HTTP_X_AUTH_USER': username, 'HTTP_X_AUTH_KEY': 'unset', 'HTTP_X_AUTH_URL': 'url'}
        if self._token_info:
            headers['keystone.token_info'] = self._token_info
        return headers