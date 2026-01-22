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
class BaseAssumeRoleCredentialFetcher(CachedCredentialFetcher):

    def __init__(self, client_creator, role_arn, extra_args=None, cache=None, expiry_window_seconds=None):
        self._client_creator = client_creator
        self._role_arn = role_arn
        if extra_args is None:
            self._assume_kwargs = {}
        else:
            self._assume_kwargs = deepcopy(extra_args)
        self._assume_kwargs['RoleArn'] = self._role_arn
        self._role_session_name = self._assume_kwargs.get('RoleSessionName')
        self._using_default_session_name = False
        if not self._role_session_name:
            self._generate_assume_role_name()
        super().__init__(cache, expiry_window_seconds)

    def _generate_assume_role_name(self):
        self._role_session_name = 'botocore-session-%s' % int(time.time())
        self._assume_kwargs['RoleSessionName'] = self._role_session_name
        self._using_default_session_name = True

    def _create_cache_key(self):
        """Create a predictable cache key for the current configuration.

        The cache key is intended to be compatible with file names.
        """
        args = deepcopy(self._assume_kwargs)
        if self._using_default_session_name:
            del args['RoleSessionName']
        if 'Policy' in args:
            args['Policy'] = json.loads(args['Policy'])
        args = json.dumps(args, sort_keys=True)
        argument_hash = sha1(args.encode('utf-8')).hexdigest()
        return self._make_file_safe(argument_hash)