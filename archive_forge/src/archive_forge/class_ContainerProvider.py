import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
class ContainerProvider(CredentialProvider):
    METHOD = 'container-role'
    CANONICAL_NAME = 'EcsContainer'
    ENV_VAR = 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI'
    ENV_VAR_FULL = 'AWS_CONTAINER_CREDENTIALS_FULL_URI'
    ENV_VAR_AUTH_TOKEN = 'AWS_CONTAINER_AUTHORIZATION_TOKEN'
    ENV_VAR_AUTH_TOKEN_FILE = 'AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE'

    def __init__(self, environ=None, fetcher=None):
        if environ is None:
            environ = os.environ
        if fetcher is None:
            fetcher = ContainerMetadataFetcher()
        self._environ = environ
        self._fetcher = fetcher

    def load(self):
        if self.ENV_VAR in self._environ or self.ENV_VAR_FULL in self._environ:
            return self._retrieve_or_fail()

    def _retrieve_or_fail(self):
        if self._provided_relative_uri():
            full_uri = self._fetcher.full_url(self._environ[self.ENV_VAR])
        else:
            full_uri = self._environ[self.ENV_VAR_FULL]
        headers = self._build_headers()
        fetcher = self._create_fetcher(full_uri, headers)
        creds = fetcher()
        return RefreshableCredentials(access_key=creds['access_key'], secret_key=creds['secret_key'], token=creds['token'], method=self.METHOD, expiry_time=_parse_if_needed(creds['expiry_time']), refresh_using=fetcher)

    def _build_headers(self):
        auth_token = None
        if self.ENV_VAR_AUTH_TOKEN_FILE in self._environ:
            auth_token_file_path = self._environ[self.ENV_VAR_AUTH_TOKEN_FILE]
            with open(auth_token_file_path) as token_file:
                auth_token = token_file.read()
        elif self.ENV_VAR_AUTH_TOKEN in self._environ:
            auth_token = self._environ[self.ENV_VAR_AUTH_TOKEN]
        if auth_token is not None:
            self._validate_auth_token(auth_token)
            return {'Authorization': auth_token}

    def _validate_auth_token(self, auth_token):
        if '\r' in auth_token or '\n' in auth_token:
            raise ValueError('Auth token value is not a legal header value')

    def _create_fetcher(self, full_uri, headers):

        def fetch_creds():
            try:
                response = self._fetcher.retrieve_full_uri(full_uri, headers=headers)
            except MetadataRetrievalError as e:
                logger.debug('Error retrieving container metadata: %s', e, exc_info=True)
                raise CredentialRetrievalError(provider=self.METHOD, error_msg=str(e))
            return {'access_key': response['AccessKeyId'], 'secret_key': response['SecretAccessKey'], 'token': response['Token'], 'expiry_time': response['Expiration']}
        return fetch_creds

    def _provided_relative_uri(self):
        return self.ENV_VAR in self._environ