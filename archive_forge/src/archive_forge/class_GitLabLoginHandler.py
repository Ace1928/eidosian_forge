import json
import os
import re
import uuid
from urllib.parse import urlencode
import tornado.auth
import tornado.gen
import tornado.web
from celery.utils.imports import instantiate
from tornado.options import options
from ..views import BaseHandler
from ..views.error import NotFoundErrorHandler
class GitLabLoginHandler(BaseHandler, tornado.auth.OAuth2Mixin):
    _OAUTH_GITLAB_DOMAIN = os.getenv('FLOWER_GITLAB_OAUTH_DOMAIN', 'gitlab.com')
    _OAUTH_AUTHORIZE_URL = f'https://{_OAUTH_GITLAB_DOMAIN}/oauth/authorize'
    _OAUTH_ACCESS_TOKEN_URL = f'https://{_OAUTH_GITLAB_DOMAIN}/oauth/token'
    _OAUTH_NO_CALLBACKS = False

    async def get_authenticated_user(self, redirect_uri, code):
        body = urlencode({'redirect_uri': redirect_uri, 'code': code, 'client_id': self.settings['oauth']['key'], 'client_secret': self.settings['oauth']['secret'], 'grant_type': 'authorization_code'})
        response = await self.get_auth_http_client().fetch(self._OAUTH_ACCESS_TOKEN_URL, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}, body=body)
        if response.error:
            raise tornado.auth.AuthError(f'OAuth authenticator error: {response}')
        return json.loads(response.body.decode('utf-8'))

    async def get(self):
        redirect_uri = self.settings['oauth']['redirect_uri']
        if self.get_argument('code', False):
            user = await self.get_authenticated_user(redirect_uri=redirect_uri, code=self.get_argument('code'))
            await self._on_auth(user)
        else:
            self.authorize_redirect(redirect_uri=redirect_uri, client_id=self.settings['oauth']['key'], scope=['read_api'], response_type='code', extra_params={'approval_prompt': ''})

    async def _on_auth(self, user):
        if not user:
            raise tornado.web.HTTPError(500, 'OAuth authentication failed')
        access_token = user['access_token']
        allowed_groups = os.environ.get('FLOWER_GITLAB_AUTH_ALLOWED_GROUPS', '')
        allowed_groups = [group.strip() for group in allowed_groups.split(',') if group]
        try:
            response = await self.get_auth_http_client().fetch(f'https://{self._OAUTH_GITLAB_DOMAIN}/api/v4/user', headers={'Authorization': 'Bearer ' + access_token, 'User-agent': 'Tornado auth'})
        except Exception as e:
            raise tornado.web.HTTPError(403, f'GitLab auth failed: {e}')
        user_email = json.loads(response.body.decode('utf-8'))['email']
        email_allowed = authenticate(self.application.options.auth, user_email)
        matching_groups = []
        if allowed_groups:
            min_access_level = os.environ.get('FLOWER_GITLAB_MIN_ACCESS_LEVEL', '20')
            response = await self.get_auth_http_client().fetch(f'https://{self._OAUTH_GITLAB_DOMAIN}/api/v4/groups?min_access_level={min_access_level}', headers={'Authorization': 'Bearer ' + access_token, 'User-agent': 'Tornado auth'})
            matching_groups = [group['id'] for group in json.loads(response.body.decode('utf-8')) if group['full_path'] in allowed_groups]
        if not email_allowed or (allowed_groups and len(matching_groups) == 0):
            message = 'Access denied. Please use another account or contact your admin.'
            raise tornado.web.HTTPError(403, message)
        self.set_secure_cookie('user', str(user_email))
        next_ = self.get_argument('next', self.application.options.url_prefix or '/')
        if self.application.options.url_prefix and next_[0] != '/':
            next_ = '/' + next_
        self.redirect(next_)