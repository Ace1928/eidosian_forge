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
class AzureAdLoginHandler(OAuthLoginHandler):
    _API_BASE_HEADERS = {'Accept': 'application/json', 'User-Agent': 'Tornado OAuth'}
    _OAUTH_ACCESS_TOKEN_URL_ = 'https://login.microsoftonline.com/{tenant}/oauth2/token'
    _OAUTH_AUTHORIZE_URL_ = 'https://login.microsoftonline.com/{tenant}/oauth2/authorize'
    _OAUTH_USER_URL_ = ''
    _USER_KEY = 'unique_name'

    @property
    def _OAUTH_ACCESS_TOKEN_URL(self):
        tenant = os.environ.get('AAD_TENANT_ID', config.oauth_extra_params.get('tenant', 'common'))
        return self._OAUTH_ACCESS_TOKEN_URL_.format(tenant=tenant)

    @property
    def _OAUTH_AUTHORIZE_URL(self):
        tenant = os.environ.get('AAD_TENANT_ID', config.oauth_extra_params.get('tenant', 'common'))
        return self._OAUTH_AUTHORIZE_URL_.format(tenant=tenant)

    @property
    def _OAUTH_USER_URL(self):
        return self._OAUTH_USER_URL_.format(**config.oauth_extra_params)