import asyncio
import base64
import codecs
import datetime as dt
import hashlib
import json
import logging
import os
import re
import urllib.parse as urlparse
import uuid
from base64 import urlsafe_b64encode
from functools import partial
import tornado
from bokeh.server.auth_provider import AuthProvider
from tornado.auth import OAuth2Mixin
from tornado.httpclient import HTTPError as HTTPClientError, HTTPRequest
from tornado.web import HTTPError, RequestHandler, decode_signed_value
from tornado.websocket import WebSocketHandler
from .config import config
from .entry_points import entry_points_for
from .io.resources import (
from .io.state import state
from .util import base64url_encode, decode_token
class BasicLoginHandler(RequestHandler):
    _login_template = BASIC_LOGIN_TEMPLATE

    def get(self):
        try:
            errormessage = self.get_argument('error')
        except Exception:
            errormessage = ''
        next_url = self.get_argument('next', None)
        if next_url:
            self.set_cookie('next_url', next_url)
        html = self._login_template.render(errormessage=errormessage, PANEL_CDN=CDN_DIST)
        self.write(html)

    def _validate(self, username, password):
        if 'basic_auth' in state._server_config.get(self.application, {}):
            auth_info = state._server_config[self.application]['basic_auth']
        else:
            auth_info = config.basic_auth
        if isinstance(auth_info, str) and os.path.isfile(auth_info):
            with open(auth_info, encoding='utf-8') as auth_file:
                auth_info = json.loads(auth_file.read())
        if isinstance(auth_info, dict):
            if username not in auth_info:
                return False
            return password == auth_info[username]
        elif password == auth_info:
            return True
        return False

    def post(self):
        username = self.get_argument('username', '')
        password = self.get_argument('password', '')
        auth = self._validate(username, password)
        if auth:
            self.set_current_user(username)
            next_url = self.get_cookie('next_url', '/')
            self.redirect(next_url)
        else:
            error_msg = '?error=' + tornado.escape.url_escape('Invalid username or password!')
            self.redirect(self.request.uri + error_msg)

    def set_current_user(self, user):
        if not user:
            self.clear_cookie('is_guest')
            self.clear_cookie('user')
            return
        self.clear_cookie('is_guest')
        self.set_secure_cookie('user', user, expires_days=config.oauth_expiry)
        id_token = base64url_encode(json.dumps({'user': user}))
        if state.encryption:
            id_token = state.encryption.encrypt(id_token.encode('utf-8'))
        self.set_secure_cookie('id_token', id_token, expires_days=config.oauth_expiry)