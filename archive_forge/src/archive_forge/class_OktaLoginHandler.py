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
class OktaLoginHandler(BaseHandler, tornado.auth.OAuth2Mixin):
    _OAUTH_NO_CALLBACKS = False
    _OAUTH_SETTINGS_KEY = 'oauth'

    @property
    def base_url(self):
        return os.environ.get('FLOWER_OAUTH2_OKTA_BASE_URL')

    @property
    def _OAUTH_AUTHORIZE_URL(self):
        return f'{self.base_url}/v1/authorize'

    @property
    def _OAUTH_ACCESS_TOKEN_URL(self):
        return f'{self.base_url}/v1/token'

    @property
    def _OAUTH_USER_INFO_URL(self):
        return f'{self.base_url}/v1/userinfo'

    async def get_access_token(self, redirect_uri, code):
        body = urlencode({'redirect_uri': redirect_uri, 'code': code, 'client_id': self.settings[self._OAUTH_SETTINGS_KEY]['key'], 'client_secret': self.settings[self._OAUTH_SETTINGS_KEY]['secret'], 'grant_type': 'authorization_code'})
        response = await self.get_auth_http_client().fetch(self._OAUTH_ACCESS_TOKEN_URL, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}, body=body)
        if response.error:
            raise tornado.auth.AuthError(f'OAuth authenticator error: {response}')
        return json.loads(response.body.decode('utf-8'))

    async def get(self):
        redirect_uri = self.settings[self._OAUTH_SETTINGS_KEY]['redirect_uri']
        if self.get_argument('code', False):
            expected_state = (self.get_secure_cookie('oauth_state') or b'').decode('utf-8')
            returned_state = self.get_argument('state')
            if returned_state is None or returned_state != expected_state:
                raise tornado.auth.AuthError('OAuth authenticator error: State tokens do not match')
            access_token_response = await self.get_access_token(redirect_uri=redirect_uri, code=self.get_argument('code'))
            await self._on_auth(access_token_response)
        else:
            state = str(uuid.uuid4())
            self.set_secure_cookie('oauth_state', state)
            self.authorize_redirect(redirect_uri=redirect_uri, client_id=self.settings[self._OAUTH_SETTINGS_KEY]['key'], scope=['openid email'], response_type='code', extra_params={'state': state})

    async def _on_auth(self, access_token_response):
        if not access_token_response:
            raise tornado.web.HTTPError(500, 'OAuth authentication failed')
        access_token = access_token_response['access_token']
        response = await self.get_auth_http_client().fetch(self._OAUTH_USER_INFO_URL, headers={'Authorization': 'Bearer ' + access_token, 'User-agent': 'Tornado auth'})
        decoded_body = json.loads(response.body.decode('utf-8'))
        email = (decoded_body.get('email') or '').strip()
        email_verified = decoded_body.get('email_verified') and authenticate(self.application.options.auth, email)
        if not email_verified:
            message = 'Access denied. Please use another account or ask your admin to add your email to flower --auth.'
            raise tornado.web.HTTPError(403, message)
        self.set_secure_cookie('user', str(email))
        self.clear_cookie('oauth_state')
        next_ = self.get_argument('next', self.application.options.url_prefix or '/')
        if self.application.options.url_prefix and next_[0] != '/':
            next_ = '/' + next_
        self.redirect(next_)