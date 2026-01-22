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
class SharedCredentialProvider(CredentialProvider):
    METHOD = 'shared-credentials-file'
    CANONICAL_NAME = 'SharedCredentials'
    ACCESS_KEY = 'aws_access_key_id'
    SECRET_KEY = 'aws_secret_access_key'
    TOKENS = ['aws_security_token', 'aws_session_token']

    def __init__(self, creds_filename, profile_name=None, ini_parser=None):
        self._creds_filename = creds_filename
        if profile_name is None:
            profile_name = 'default'
        self._profile_name = profile_name
        if ini_parser is None:
            ini_parser = botocore.configloader.raw_config_parse
        self._ini_parser = ini_parser

    def load(self):
        try:
            available_creds = self._ini_parser(self._creds_filename)
        except ConfigNotFound:
            return None
        if self._profile_name in available_creds:
            config = available_creds[self._profile_name]
            if self.ACCESS_KEY in config:
                logger.info('Found credentials in shared credentials file: %s', self._creds_filename)
                access_key, secret_key = self._extract_creds_from_mapping(config, self.ACCESS_KEY, self.SECRET_KEY)
                token = self._get_session_token(config)
                return Credentials(access_key, secret_key, token, method=self.METHOD)

    def _get_session_token(self, config):
        for token_envvar in self.TOKENS:
            if token_envvar in config:
                return config[token_envvar]