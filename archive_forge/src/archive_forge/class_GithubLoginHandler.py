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
class GithubLoginHandler(BaseHandler, tornado.auth.OAuth2Mixin):
    _OAUTH_DOMAIN = os.getenv('FLOWER_GITHUB_OAUTH_DOMAIN', 'github.com')
    _OAUTH_AUTHORIZE_URL = f'https://{_OAUTH_DOMAIN}/login/oauth/authorize'
    _OAUTH_ACCESS_TOKEN_URL = f'https://{_OAUTH_DOMAIN}/login/oauth/access_token'
    _OAUTH_NO_CALLBACKS = False
    _OAUTH_SETTINGS_KEY = 'oauth'

    async def get_authenticated_user(self, redirect_uri, code):
        body = urlencode({'redirect_uri': redirect_uri, 'code': code, 'client_id': self.settings[self._OAUTH_SETTINGS_KEY]['key'], 'client_secret': self.settings[self._OAUTH_SETTINGS_KEY]['secret'], 'grant_type': 'authorization_code'})
        response = await self.get_auth_http_client().fetch(self._OAUTH_ACCESS_TOKEN_URL, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}, body=body)
        if response.error:
            raise tornado.auth.AuthError(f'OAuth authenticator error: {response}')
        return json.loads(response.body.decode('utf-8'))

    async def get(self):
        redirect_uri = self.settings[self._OAUTH_SETTINGS_KEY]['redirect_uri']
        if self.get_argument('code', False):
            user = await self.get_authenticated_user(redirect_uri=redirect_uri, code=self.get_argument('code'))
            await self._on_auth(user)
        else:
            self.authorize_redirect(redirect_uri=redirect_uri, client_id=self.settings[self._OAUTH_SETTINGS_KEY]['key'], scope=['user:email'], response_type='code', extra_params={'approval_prompt': ''})

    async def _on_auth(self, user):
        if not user:
            raise tornado.web.HTTPError(500, 'OAuth authentication failed')
        access_token = user['access_token']
        response = await self.get_auth_http_client().fetch(f'https://api.{self._OAUTH_DOMAIN}/user/emails', headers={'Authorization': 'token ' + access_token, 'User-agent': 'Tornado auth'})
        emails = [email['email'].lower() for email in json.loads(response.body.decode('utf-8')) if email['verified'] and authenticate(self.application.options.auth, email['email'])]
        if not emails:
            message = 'Access denied. Please use another account or ask your admin to add your email to flower --auth.'
            raise tornado.web.HTTPError(403, message)
        self.set_secure_cookie('user', str(emails.pop()))
        next_ = self.get_argument('next', self.application.options.url_prefix or '/')
        if self.application.options.url_prefix and next_[0] != '/':
            next_ = '/' + next_
        self.redirect(next_)