import atexit
import base64
import copy
import datetime
import json
import logging
import os
import platform
import tempfile
import time
import google.auth
import google.auth.transport.requests
import oauthlib.oauth2
import urllib3
from ruamel import yaml
from requests_oauthlib import OAuth2Session
from six import PY3
from kubernetes.client import ApiClient, Configuration
from kubernetes.config.exec_provider import ExecProvider
from .config_exception import ConfigException
from .dateutil import UTC, format_rfc3339, parse_rfc3339
def _load_oid_token(self, provider):
    if 'config' not in provider:
        return
    reserved_characters = frozenset(['=', '+', '/'])
    token = provider['config']['id-token']
    if any((char in token for char in reserved_characters)):
        return
    parts = token.split('.')
    if len(parts) != 3:
        return
    padding = (4 - len(parts[1]) % 4) * '='
    if len(padding) == 3:
        return
    if PY3:
        jwt_attributes = json.loads(base64.b64decode(parts[1] + padding).decode('utf-8'))
    else:
        jwt_attributes = json.loads(base64.b64decode(parts[1] + padding))
    expire = jwt_attributes.get('exp')
    if expire is not None and _is_expired(datetime.datetime.fromtimestamp(expire, tz=UTC)):
        self._refresh_oidc(provider)
        if self._config_persister:
            self._config_persister(self._config.value)
    self.token = 'Bearer %s' % provider['config']['id-token']
    return self.token