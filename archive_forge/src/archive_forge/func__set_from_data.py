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
def _set_from_data(self, data):
    expected_keys = ['access_key', 'secret_key', 'token', 'expiry_time']
    if not data:
        missing_keys = expected_keys
    else:
        missing_keys = [k for k in expected_keys if k not in data]
    if missing_keys:
        message = 'Credential refresh failed, response did not contain: %s'
        raise CredentialRetrievalError(provider=self.method, error_msg=message % ', '.join(missing_keys))
    self.access_key = data['access_key']
    self.secret_key = data['secret_key']
    self.token = data['token']
    self._expiry_time = parse(data['expiry_time'])
    logger.debug('Retrieved credentials will expire at: %s', self._expiry_time)
    self._normalize()