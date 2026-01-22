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
def _on_auth(self, id_token, access_token, refresh_token=None, expires_in=None):
    if isinstance(id_token, str):
        decoded = decode_token(id_token)
    else:
        decoded = id_token
        id_token = base64url_encode(json.dumps(id_token))
    user_key = config.oauth_jwt_user or self._USER_KEY
    if user_key in decoded:
        user = decoded[user_key]
    else:
        log.error('%s token payload did not contain expected %r.', type(self).__name__, user_key)
        raise HTTPError(401, 'OAuth token payload missing user information')
    self.clear_cookie('is_guest')
    self.set_secure_cookie('user', user, expires_days=config.oauth_expiry)
    if state.encryption:
        access_token = state.encryption.encrypt(access_token.encode('utf-8'))
        id_token = state.encryption.encrypt(id_token.encode('utf-8'))
        if refresh_token:
            refresh_token = state.encryption.encrypt(refresh_token.encode('utf-8'))
    self.set_secure_cookie('access_token', access_token, expires_days=config.oauth_expiry)
    self.set_secure_cookie('id_token', id_token, expires_days=config.oauth_expiry)
    if expires_in:
        now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
        self.set_secure_cookie('oauth_expiry', str(int(now_ts + expires_in)), expires_days=config.oauth_expiry)
    if refresh_token:
        self.set_secure_cookie('refresh_token', refresh_token, expires_days=config.oauth_expiry)
    if user in state._oauth_user_overrides:
        state._oauth_user_overrides.pop(user, None)
    return user