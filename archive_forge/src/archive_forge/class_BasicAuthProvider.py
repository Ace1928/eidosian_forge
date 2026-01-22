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
class BasicAuthProvider(AuthProvider):
    """
    An AuthProvider which serves a simple login and logout page.
    """

    def __init__(self, login_endpoint=None, logout_endpoint=None, login_template=None, logout_template=None, error_template=None, guest_endpoints=None):
        if error_template is None:
            self._error_template = ERROR_TEMPLATE
        else:
            with open(error_template) as f:
                self._error_template = _env.from_string(f.read())
        if logout_template is None:
            self._logout_template = LOGOUT_TEMPLATE
        else:
            with open(logout_template) as f:
                self._logout_template = _env.from_string(f.read())
        if login_template is None:
            self._login_template = BASIC_LOGIN_TEMPLATE
        else:
            with open(login_template) as f:
                self._login_template = _env.from_string(f.read())
        self._login_endpoint = login_endpoint or '/login'
        self._logout_endpoint = logout_endpoint or '/logout'
        self._guest_endpoints = guest_endpoints or []
        state.on_session_destroyed(self._remove_user)
        super().__init__()

    def _remove_user(self, session_context):
        guest_cookie = session_context.request.cookies.get('is_guest')
        user_cookie = session_context.request.cookies.get('user')
        if guest_cookie:
            user = 'guest'
        elif user_cookie:
            user = decode_signed_value(config.cookie_secret, 'user', user_cookie)
            if user:
                user = user.decode('utf-8')
        else:
            user = None
        if not user:
            return
        state._active_users[user] -= 1
        if not state._active_users[user]:
            del state._active_users[user]

    def _allow_guest(self, uri):
        if config.oauth_optional and (not (uri == self._login_endpoint or '?code=' in uri)):
            return True
        return True if uri.replace('/ws', '') in self._guest_endpoints else False

    @property
    def get_user(self):

        def get_user(request_handler):
            user = request_handler.get_secure_cookie('user', max_age_days=config.oauth_expiry)
            if user:
                user = user.decode('utf-8')
            elif self._allow_guest(request_handler.request.uri):
                user = 'guest'
                request_handler.request.cookies['is_guest'] = '1'
                if not isinstance(request_handler, WebSocketHandler):
                    request_handler.set_cookie('is_guest', '1', expires_days=config.oauth_expiry)
            if user and isinstance(request_handler, WebSocketHandler):
                state._active_users[user] += 1
            return user
        return get_user

    @property
    def login_url(self):
        return self._login_endpoint

    @property
    def login_handler(self):
        BasicLoginHandler._login_endpoint = self._login_endpoint
        BasicLoginHandler._login_template = self._login_template
        return BasicLoginHandler

    @property
    def logout_url(self):
        return self._logout_endpoint

    @property
    def logout_handler(self):
        if self._logout_template:
            LogoutHandler._logout_template = self._logout_template
        LogoutHandler._login_endpoint = self._login_endpoint
        return LogoutHandler