import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
class GoogleOAuth2Credential:
    default_credential_file = '~/.google_libcloud_auth'

    def __init__(self, user_id, key, auth_type=None, credential_file=None, scopes=None, **kwargs):
        self.auth_type = auth_type or GoogleAuthType.guess_type(user_id)
        if self.auth_type not in GoogleAuthType.ALL_TYPES:
            raise GoogleAuthError('Invalid auth type: %s' % self.auth_type)
        if not GoogleAuthType.is_oauth2(self.auth_type):
            raise GoogleAuthError('Auth type %s cannot be used with OAuth2' % self.auth_type)
        self.user_id = user_id
        self.key = key
        default_credential_file = '.'.join([self.default_credential_file, user_id])
        self.credential_file = credential_file or default_credential_file
        self.scopes = scopes or ['https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/devstorage.full_control', 'https://www.googleapis.com/auth/ndev.clouddns.readwrite']
        self.token = self._get_token_from_file()
        if self.auth_type == GoogleAuthType.GCE:
            self.oauth2_conn = GoogleGCEServiceAcctAuthConnection(self.user_id, self.scopes, **kwargs)
        elif self.auth_type == GoogleAuthType.SA:
            self.oauth2_conn = GoogleServiceAcctAuthConnection(self.user_id, self.key, self.scopes, **kwargs)
        elif self.auth_type == GoogleAuthType.IA:
            self.oauth2_conn = GoogleInstalledAppAuthConnection(self.user_id, self.key, self.scopes, **kwargs)
        else:
            raise GoogleAuthError('Invalid auth_type: %s' % str(self.auth_type))
        if self.token is None:
            self.token = self.oauth2_conn.get_new_token()
            self._write_token_to_file()

    @property
    def access_token(self):
        if self.token_expire_utc_datetime < _utcnow():
            self._refresh_token()
        return self.token['access_token']

    @property
    def token_expire_utc_datetime(self):
        return _from_utc_timestamp(self.token['expire_time'])

    def _refresh_token(self):
        self.token = self.oauth2_conn.refresh_token(self.token)
        self._write_token_to_file()

    def _get_token_from_file(self):
        """
        Read credential file and return token information.
        Mocked in libcloud.test.common.google.GoogleTestCase.

        :return:  Token information dictionary, or None
        :rtype:   ``dict`` or ``None``
        """
        token = None
        filename = os.path.realpath(os.path.expanduser(self.credential_file))
        try:
            with open(filename) as f:
                data = f.read()
            token = json.loads(data)
        except (OSError, ValueError) as e:
            LOG.info('Failed to read cached auth token from file "%s": %s', filename, str(e))
        return token

    def _write_token_to_file(self):
        """
        Write token to credential file.
        Mocked in libcloud.test.common.google.GoogleTestCase.
        """
        filename = os.path.expanduser(self.credential_file)
        filename = os.path.realpath(filename)
        try:
            data = json.dumps(self.token)
            write_flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
            with os.fdopen(os.open(filename, write_flags, int('600', 8)), 'w') as f:
                f.write(data)
        except Exception as e:
            LOG.info('Failed to write auth token to file "%s": %s', filename, str(e))