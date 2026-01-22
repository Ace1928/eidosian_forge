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
def _protected_refresh(self, is_mandatory):
    try:
        metadata = self._refresh_using()
    except Exception:
        period_name = 'mandatory' if is_mandatory else 'advisory'
        logger.warning('Refreshing temporary credentials failed during %s refresh period.', period_name, exc_info=True)
        if is_mandatory:
            raise
        return
    self._set_from_data(metadata)
    self._frozen_credentials = ReadOnlyCredentials(self._access_key, self._secret_key, self._token)
    if self._is_expired():
        msg = 'Credentials were refreshed, but the refreshed credentials are still expired.'
        logger.warning(msg)
        raise RuntimeError(msg)