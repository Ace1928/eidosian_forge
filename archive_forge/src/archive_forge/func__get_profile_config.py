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
def _get_profile_config(self, key):
    if self._profile_config is None:
        loaded_config = self._load_config()
        profiles = loaded_config.get('profiles', {})
        self._profile_config = profiles.get(self._profile_name, {})
    return self._profile_config.get(key)