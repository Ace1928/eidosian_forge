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
def _retrieve_or_fail(self):
    if self._provided_relative_uri():
        full_uri = self._fetcher.full_url(self._environ[self.ENV_VAR])
    else:
        full_uri = self._environ[self.ENV_VAR_FULL]
    headers = self._build_headers()
    fetcher = self._create_fetcher(full_uri, headers)
    creds = fetcher()
    return RefreshableCredentials(access_key=creds['access_key'], secret_key=creds['secret_key'], token=creds['token'], method=self.METHOD, expiry_time=_parse_if_needed(creds['expiry_time']), refresh_using=fetcher)