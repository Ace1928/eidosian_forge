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
def _resolve_sso_session_reference(self, profile_config, sso_sessions):
    sso_session_name = profile_config.get('sso_session')
    if sso_session_name is None:
        return (profile_config, ())
    if sso_session_name not in sso_sessions:
        error_msg = f'The specified sso-session does not exist: "{sso_session_name}"'
        raise InvalidConfigError(error_msg=error_msg)
    config = profile_config.copy()
    session = sso_sessions[sso_session_name]
    for config_var, val in session.items():
        if config.get(config_var, val) != val:
            error_msg = f'The value for {config_var} is inconsistent between profile ({config[config_var]}) and sso-session ({val}).'
            raise InvalidConfigError(error_msg=error_msg)
        config[config_var] = val
    return (config, ('sso_session',))