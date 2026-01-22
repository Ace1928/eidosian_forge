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
class AssumeRoleWithWebIdentityProvider(CredentialProvider):
    METHOD = 'assume-role-with-web-identity'
    CANONICAL_NAME = None
    _CONFIG_TO_ENV_VAR = {'web_identity_token_file': 'AWS_WEB_IDENTITY_TOKEN_FILE', 'role_session_name': 'AWS_ROLE_SESSION_NAME', 'role_arn': 'AWS_ROLE_ARN'}

    def __init__(self, load_config, client_creator, profile_name, cache=None, disable_env_vars=False, token_loader_cls=None):
        self.cache = cache
        self._load_config = load_config
        self._client_creator = client_creator
        self._profile_name = profile_name
        self._profile_config = None
        self._disable_env_vars = disable_env_vars
        if token_loader_cls is None:
            token_loader_cls = FileWebIdentityTokenLoader
        self._token_loader_cls = token_loader_cls

    def load(self):
        return self._assume_role_with_web_identity()

    def _get_profile_config(self, key):
        if self._profile_config is None:
            loaded_config = self._load_config()
            profiles = loaded_config.get('profiles', {})
            self._profile_config = profiles.get(self._profile_name, {})
        return self._profile_config.get(key)

    def _get_env_config(self, key):
        if self._disable_env_vars:
            return None
        env_key = self._CONFIG_TO_ENV_VAR.get(key)
        if env_key and env_key in os.environ:
            return os.environ[env_key]
        return None

    def _get_config(self, key):
        env_value = self._get_env_config(key)
        if env_value is not None:
            return env_value
        return self._get_profile_config(key)

    def _assume_role_with_web_identity(self):
        token_path = self._get_config('web_identity_token_file')
        if not token_path:
            return None
        token_loader = self._token_loader_cls(token_path)
        role_arn = self._get_config('role_arn')
        if not role_arn:
            error_msg = 'The provided profile or the current environment is configured to assume role with web identity but has no role ARN configured. Ensure that the profile has the role_arnconfiguration set or the AWS_ROLE_ARN env var is set.'
            raise InvalidConfigError(error_msg=error_msg)
        extra_args = {}
        role_session_name = self._get_config('role_session_name')
        if role_session_name is not None:
            extra_args['RoleSessionName'] = role_session_name
        fetcher = AssumeRoleWithWebIdentityCredentialFetcher(client_creator=self._client_creator, web_identity_token_loader=token_loader, role_arn=role_arn, extra_args=extra_args, cache=self.cache)
        return DeferredRefreshableCredentials(method=self.METHOD, refresh_using=fetcher.fetch_credentials)