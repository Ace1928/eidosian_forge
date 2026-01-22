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
class SSOCredentialFetcher(CachedCredentialFetcher):
    _UTC_DATE_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

    def __init__(self, start_url, sso_region, role_name, account_id, client_creator, token_loader=None, cache=None, expiry_window_seconds=None, token_provider=None, sso_session_name=None):
        self._client_creator = client_creator
        self._sso_region = sso_region
        self._role_name = role_name
        self._account_id = account_id
        self._start_url = start_url
        self._token_loader = token_loader
        self._token_provider = token_provider
        self._sso_session_name = sso_session_name
        super().__init__(cache, expiry_window_seconds)

    def _create_cache_key(self):
        """Create a predictable cache key for the current configuration.

        The cache key is intended to be compatible with file names.
        """
        args = {'roleName': self._role_name, 'accountId': self._account_id}
        if self._sso_session_name:
            args['sessionName'] = self._sso_session_name
        else:
            args['startUrl'] = self._start_url
        args = json.dumps(args, sort_keys=True, separators=(',', ':'))
        argument_hash = sha1(args.encode('utf-8')).hexdigest()
        return self._make_file_safe(argument_hash)

    def _parse_timestamp(self, timestamp_ms):
        timestamp_seconds = timestamp_ms / 1000.0
        timestamp = datetime.datetime.fromtimestamp(timestamp_seconds, tzutc())
        return timestamp.strftime(self._UTC_DATE_FORMAT)

    def _get_credentials(self):
        """Get credentials by calling SSO get role credentials."""
        config = Config(signature_version=UNSIGNED, region_name=self._sso_region)
        client = self._client_creator('sso', config=config)
        if self._token_provider:
            initial_token_data = self._token_provider.load_token()
            token = initial_token_data.get_frozen_token().token
        else:
            token = self._token_loader(self._start_url)['accessToken']
        kwargs = {'roleName': self._role_name, 'accountId': self._account_id, 'accessToken': token}
        try:
            response = client.get_role_credentials(**kwargs)
        except client.exceptions.UnauthorizedException:
            raise UnauthorizedSSOTokenError()
        credentials = response['roleCredentials']
        credentials = {'ProviderType': 'sso', 'Credentials': {'AccessKeyId': credentials['accessKeyId'], 'SecretAccessKey': credentials['secretAccessKey'], 'SessionToken': credentials['sessionToken'], 'Expiration': self._parse_timestamp(credentials['expiration'])}}
        return credentials