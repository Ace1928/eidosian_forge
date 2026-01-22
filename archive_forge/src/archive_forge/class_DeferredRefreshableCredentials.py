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
class DeferredRefreshableCredentials(RefreshableCredentials):
    """Refreshable credentials that don't require initial credentials.

    refresh_using will be called upon first access.
    """

    def __init__(self, refresh_using, method, time_fetcher=_local_now):
        self._refresh_using = refresh_using
        self._access_key = None
        self._secret_key = None
        self._token = None
        self._expiry_time = None
        self._time_fetcher = time_fetcher
        self._refresh_lock = threading.Lock()
        self.method = method
        self._frozen_credentials = None

    def refresh_needed(self, refresh_in=None):
        if self._frozen_credentials is None:
            return True
        return super().refresh_needed(refresh_in)