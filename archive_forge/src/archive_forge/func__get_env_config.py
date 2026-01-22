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
def _get_env_config(self, key):
    if self._disable_env_vars:
        return None
    env_key = self._CONFIG_TO_ENV_VAR.get(key)
    if env_key and env_key in os.environ:
        return os.environ[env_key]
    return None