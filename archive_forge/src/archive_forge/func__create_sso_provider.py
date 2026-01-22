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
def _create_sso_provider(self, profile_name):
    return SSOProvider(load_config=lambda: self._session.full_config, client_creator=self._session.create_client, profile_name=profile_name, cache=self._cache, token_cache=self._sso_token_cache, token_provider=SSOTokenProvider(self._session, cache=self._sso_token_cache, profile_name=profile_name))