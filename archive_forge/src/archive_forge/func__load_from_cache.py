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
def _load_from_cache(self):
    if self._cache_key in self._cache:
        creds = deepcopy(self._cache[self._cache_key])
        if not self._is_expired(creds):
            return creds
        else:
            logger.debug('Credentials were found in cache, but they are expired.')
    return None